use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, matmul_transb, rms_norm, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

use std::vec;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;

        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding Lookup
        // input 与词嵌入矩阵相乘得到残差

        OP::gather(&mut residual, input, &self.params.embedding_table);

        // 这里就可以理解为 一个输入的token id 通过embedding table 得到一个残差，这个残差是128维的
        // 只看第一层
        for layer in 0..self.n_layers {
            // x = rms_norm(residual) 残差进行归一化处理 传递到隐藏层
            OP::rms_norm(
                &mut hidden_states,            // 6*128
                &residual,                     // 6*128
                &self.params.rms_att_w[layer], // 128
                self.eps,
            );

            // q 是序列长度 , n_q_h * dqkv n_q_h 是q头的数量 dqkv是q头的长度
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)

            // 隐藏层与权重矩阵相乘
            //hidden_states seq_len, self.d      wq ：n_heads * head_size
            // 所以：q: seq_len, n_q_h * dqkv dqkv*n_q_h = d
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            //hidden_states seq_len, self.d      wk ：n_kv_heads * head_size
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            // 经过验证 matmul_transb是对的
            //Q = RoPE(x @ Q_weight.T)
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            // 对q重新进行reshape
            let q = q.reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
            //K = RoPE(x @ K_weight.T)
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
                                                       // 自注意力层
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            // o_proj matmul and add residual
            OP::matmul_transb(
                &mut residual,
                1., // 这里写1 就是加上原来的值
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
            // mlp(...)
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }
    /// 生成文本
    /// token_ids: 输入的token id
    /// max_len: 生成文本的最大长度
    /// top_p: top p采样 含义：在累积概率大于top_p时停止采样
    /// top_k: top k采样 含义：在概率最大的top_k个token中采样
    /// temperature: 温度
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        // 创建一个缓存
        let mut cache = self.new_cache();
        // shape的阶数是1 那么就是一个向量
        let mut input = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        while result.len() < max_len {
            // 将input转换为tensor 每一次的输入都带着上一次的输出
            let logits = self.forward(&input, &mut cache);
            // 选取一个合适的token id
            let token_id = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(token_id);
            // 如果是结束符号，停止生成
            if token_id == self.eos_token_id {
                break;
            }
            input = Tensor::new(vec![token_id], &vec![1]);
        }
        result
    }

    pub fn chat<F: Fn(u32) -> ()>(
        &self,
        kv_cache: &mut KVCache<f32>,
        token_ids: &[u32],
        top_p: f32,
        top_k: u32,
        temperature: f32,
        output: F,
    ) -> Vec<u32> {
        // shape的阶数是1 那么就是一个向量
        let mut input = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let mut result = Vec::<u32>::new();
        loop {
            // 将input转换为tensor 每一次的输入都带着上一次的输出
            let logits = self.forward(&input, kv_cache);
            // 选取一个合适的token id
            let token_id = OP::random_sample(&logits, top_p, top_k, temperature);
            output(token_id);
            result.push(token_id);
            // 如果是结束符号，停止生成
            if token_id == self.eos_token_id {
                break;
            }
            input = Tensor::new(vec![token_id], &vec![1]);
        }
        result
    }
}
/// 自注意力层
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv) -> (n_q_h, seq * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv) -> (n_kv_h, total_seq * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,                   // n_kv_h = 4
    n_groups: usize,                 // n_q_h / n_kv_h = 2
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // 从q和k中将能进行计算的拆出来
    // 第一步将q 由seq , n_kv_h * n_groups * dqkv 转换为 n_kv_h * n_groups * dqkv , seq
    let mut q = OP::transpose_last_2d(q);
    // 对Q进行reshape
    let q = q.reshape(&vec![n_kv_h, n_groups, dqkv, seq_len]);
    // 转成n_kv_h, n_groups, seq_len，dqkv
    let q = OP::transpose_last_2d(q);

    // 同样k也是需要的 -> total_seq, n_kv_h * dqkv
    let mut k = OP::transpose_last_2d(k);
    // n_kv_h * dqkv , total_seq
    let k = k.reshape(&vec![n_kv_h, dqkv, total_seq_len]);
    // n_kv_h, total_seq, dqkv
    let k = OP::transpose_last_2d(k);

    // 对shape的最后两个做交换
    // 第二步将k 由total_seq * n_kv_h * dqkv 转换为 n_kv_h * n_groups * seq * dqkv
    // 1、计算att_scores
    for i in 0..n_kv_h {
        for j in 0..n_groups {
            // n_kv_h, n_groups, seq_len，dqkv
            let q_start = i * n_groups * seq_len * dqkv + j * seq_len * dqkv;
            // n_kv_h, total_seq, dqkv
            let k_start = i * total_seq_len * dqkv;
            let mut att_scores_ij = att_scores.slice(
                i * n_groups * seq_len * total_seq_len + j * seq_len * total_seq_len,
                &vec![seq_len, total_seq_len],
            );
            // q @ k^T
            matmul_transb(
                &mut att_scores_ij,
                0.,
                &q.slice(q_start, &vec![seq_len, dqkv]),
                &k.slice(k_start, &vec![total_seq_len, dqkv]),
                1.0,
            );
            // scale
            let att_scores_ij_data = unsafe { att_scores_ij.data_mut() };
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    att_scores_ij_data[i * total_seq_len + j] /= (dqkv as f32).sqrt();
                }
            }
        }
    }
    // 2、计算att_scores的softmax
    OP::masked_softmax(att_scores);
    // 3、计算v
    // 转换v矩阵
    let mut v = OP::transpose_last_2d(v);
    // n_kv_h * dqkv, total_seq
    let v = v.reshape(&vec![n_kv_h, dqkv, total_seq_len]);

    // 这个的作用就是先将分数与v计算出来
    // (n_kv_h, n_groups, seq, total_seq)
    // 乘 n_kv_h, total_seq, dqkv
    // 所以hidden_states_temp得到的shape就是(n_kv_h, n_groups, seq, dqkv)
    let hidden_states_temp = Tensor::new(
        hidden_states.data().to_vec(),
        &vec![n_kv_h, n_groups, seq_len, dqkv],
    );
    for i in 0..n_kv_h {
        for j in 0..n_groups {
            let v_start = i * total_seq_len * dqkv;
            let v_ij = v.slice(v_start, &vec![dqkv, total_seq_len]);
            let att_scores_ij = att_scores.slice(
                i * n_groups * seq_len * total_seq_len + j * seq_len * total_seq_len,
                &vec![seq_len, total_seq_len],
            );
            let mut hidden_states_temp_ij = hidden_states_temp.slice(
                i * n_groups * seq_len * dqkv + j * seq_len * dqkv,
                &vec![seq_len, dqkv],
            );
            // att_scores_ij @ v
            matmul_transb(&mut hidden_states_temp_ij, 0., &att_scores_ij, &v_ij, 1.0);
        }
    }
    // temp 对了以后就将temp转换为hidden_states
    // n_kv_h, n_groups, seq_len, dqkv -> (seq, n_kv_h * n_groups * dqkv)
    // transpose_last_2d 一次 n_kv_h, n_groups,dqkv,seq_len
    let mut temp = OP::transpose_last_2d(&hidden_states_temp);
    // reshape一下
    let temp = temp.reshape(&vec![n_kv_h * n_groups * dqkv, seq_len]);
    // 然后再转置一下 n_kv_h * n_groups * dqkv, seq_len -> seq_len, n_kv_h * n_groups * dqkv
    let temp = OP::transpose_last_2d(&temp);
    // 将temp的数据拷贝到hidden_states
    let temp_data = temp.data();
    unsafe {
        let hidden_states_data = hidden_states.data_mut();
        for i in 0..hidden_states_data.len() {
            hidden_states_data[i] = temp_data[i];
        }
    }
}
/// mlp是什么 前反馈神经网络
fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // 实现mlp
    // 归一化
    //hidden = rms_norm(residual)
    rms_norm(hidden_states, residual, rms_w, eps);
    //gate = hidden @ gate_weight.T
    matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    //up = hidden @ up_weight.T
    matmul_transb(up, 0., hidden_states, w_up, 1.0);
    //act = gate * sigmoid(gate) * up ## SwiGLU
    let mut act = Tensor::new(vec![1.; gate.size()], gate.shape());
    swiglu(&mut act, gate);
    let act_date = unsafe { act.data_mut() };
    for i in 0..up.size() {
        act_date[i] *= up.data()[i];
    }
    //output = act @ down_weight.T
    let mut output = Tensor::default(&vec![act.shape()[0], w_down.shape()[0]]);
    matmul_transb(&mut output, 0., &act, w_down, 1.0);
    //residual = residual + output
    let residual_date = unsafe { residual.data_mut() };
    for i in 0..residual_date.len() {
        residual_date[i] += output.data()[i];
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );
    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
