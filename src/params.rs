use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // impl 只能静态绑定，不能动态绑定
        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| -> Tensor<f32> {
            // 查找有哪些tensor
            // let _ = &safetensor
            //     .tensors()
            //     .iter()
            //     .map(|(k, _)| k)
            //     .for_each(|k| println!("{}", k));
            let tensor_view = &safetensor.tensor(name).unwrap();
            let shape = tensor_view.shape().to_vec();
            // 因为所有的都是f32，所以可以直接转换
            let data = unsafe {
                std::slice::from_raw_parts(
                    tensor_view.data().as_ptr() as *const f32,
                    tensor_view.data().len() / 4,
                )
            };
            Tensor::new(data.to_vec(), &shape)
        });
        let lm_head = get_tensor("lm_head.weight");
        let embedding_table;
        // 当"tie_word_embeddings"属性被打开时，模型最开始以及最后的embedding矩阵数据相同
        if config.tie_word_embeddings {
            embedding_table = Tensor::new(lm_head.data().to_vec(), lm_head.shape());
        } else {
            embedding_table = Tensor::default(lm_head.shape());
        }
        Self {
            embedding_table,
            rms_att_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                .collect(),
            wq: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
            rms_ffn_w: (0..config.num_hidden_layers)
                .map(|i| {
                    get_tensor(&format!(
                        "model.layers.{}.post_attention_layernorm.weight",
                        i
                    ))
                })
                .collect(),
            w_up: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            w_down: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head,
        }
    }
}
