from transformers import AutoModelForCausalLM, AutoTokenizer

class Player:
    def __init__(self,player_id,word,model) :
            self.player_id = player_id
            self.word = word
            self.model = model
    
    def __str__(self):
          return f"玩家信息：(player_id={self.player_id},word={self.word},分到的模型是{self.model})"


class local_model:
    def __init__(self,model_name,device_map):
        self.model_name = model_name
        self.device_map = f"cuda:{device_map}"
        self.model,self.tokenizer = self.load_model()
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
        print("模型与分词器加载完成")
        return model,tokenizer
    
    def chat_with_model(self,user_input):
        
        messages = [
            {"role":"user","content":user_input}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text],return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
