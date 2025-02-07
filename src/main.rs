mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use minijinja::{context, Environment};
use serde_json::Value;
use std::{fs::File, path::PathBuf, rc::Rc};
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let mut input = String::new();
    println!("请输入机器人类型 (story / chat):");

    std::io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();
    if input == "story" {
        println!("欢迎使用故事生成器，故事正在生成中...");
        let model_dir = PathBuf::from(project_dir).join("models").join("story");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        print!("\n{}", input);
        let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.);
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    } else if input == "chat" {
        println!("欢迎使用聊天机器人");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer_config_file = File::open(model_dir.join("tokenizer_config.json")).unwrap();
        let tokenizer_config: Value = serde_json::from_reader(tokenizer_config_file).unwrap();
        let tokenizer = Rc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
        let mut kv_cache = llama.new_cache();
        let mut env = Environment::new();
        env.add_template(
            "chat_template",
            tokenizer_config["chat_template"].as_str().unwrap(),
        )
        .unwrap();
        loop {
            println!("please input:");
            let mut chat_input = String::new();
            std::io::stdin().read_line(&mut chat_input).unwrap();
            let tmp = env.get_template("chat_template").unwrap();
            let input_real_value = tmp
                .render(context! {
                        messages => vec![
                            context! {
                                role=> "user",
                                content => chat_input.trim(),
                            },
                        ],
                        add_generation_prompt => true,
                })
                .unwrap();
            let binding = tokenizer.encode(input_real_value, true).unwrap();
            // 通过Jinja2模板引擎，将输入的文本转换成token id
            let input_ids = binding.get_ids();
            //let tokenizer_copy = tokenizer.clone();
            let result = llama.chat(&mut kv_cache, input_ids, 0.8, 30, 1., |_token| {
                // 这里想做成生成一个词就输出一个词
                //print!("{} ", tokenizer_copy.decode(&vec![token], true).unwrap())
            });
            print!("AI: {} ", tokenizer.decode(&result, true).unwrap());
            println!();
        }
    }
}
