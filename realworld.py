import time
from tqdm import tqdm
import pandas as pd
import json
import multiprocessing
import os
import numpy as np
import argparse
import re
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from tqdm.auto import tqdm
from trl import DPOTrainer, SFTTrainer
import torch.nn.functional as F

huggingface_token = ''
model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, token=huggingface_token,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4')
)

base_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def llama_response(role, messages):
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False) + f"<|start_header_id|>{role}<|end_header_id|>\n"
    outputs = base_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.1,
    )
    return outputs[0]["generated_text"][len(prompt):]

def llama_output(role, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False) + f"<|start_header_id|>{role}<|end_header_id|>\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model(input_ids=input_ids) 
        logits = output.logits

    logits = logits.squeeze(0)[-1, :]
    probabilities = F.softmax(logits, dim=-1)

    top_k_log_probs, top_k_token_ids = torch.topk(probabilities, k=10, dim=-1)

    token_probs = {}

    for rank, (log_prob, token_id) in enumerate(zip(top_k_log_probs, top_k_token_ids)):
        token = tokenizer.decode([token_id.item()])  # Convert token ID back to string
        token_probs[token] = log_prob.item()
    
    return token_probs

class Game:
    def __init__(
        self, real_world_setting, num_rounds, num_conversation_rounds, include_moderator
    ):

        self.num_rounds = num_rounds
        self.num_conversation_rounds = num_conversation_rounds
        self.include_moderator = include_moderator

        self.actions = ["J", "F"]

        self.real_world_setting = real_world_setting
    
        self.utility = self.real_world_setting["utility"]
        self.feedback = self.real_world_setting["feedback"]

        self.agents = ["A", "B", "moderator"]

        self.real_world_agents = ['agent1', 'agent2']
        self.real_world_actions = ['<option_A>', '<option_B>']

        self.utility_matrix = [
            [[self.utility[f"({k}, {j})"][i] for i in self.real_world_agents] for j in self.real_world_actions] for k in self.real_world_actions
        ]

        self.feedback_matrix = [
            [[self.feedback[f"({k}, {j})"][i] for i in self.real_world_agents] for j in self.real_world_actions] for k in self.real_world_actions
        ]
        
        self.evaluation = {
            agent: [
                {
                    "role": "user",
                    "content": f"""You are player {agent}.

                    Based on your conversation and the descisions, evaluate player {self.agents[1-i]} on the following criteria (score 1-10):
                    - Trust (T): Rate how truthful and reliable their statements are
                    - Cooperation (C): Rate their willingness to work together and contribute
                    - Communication (CS): Rate their clarity and effectiveness in conveying ideas 
                    - Respect (R): Rate their level of courtesy and consideration
                    - Consistency (S): Rate how well their actions align with their stated intentions
                    Give your response EXACTLY in this format:
                    <T>score</T><C>score</C><CS>score</CS><R>score</R><S>score</S>
                    """                
                }
            ]
            for i,agent in enumerate(self.agents[:2])
        }

        self.gameDescription = {agent: "" for agent in self.agents}

        self.messages = []

        for idx, agent in enumerate(self.agents):
            if agent == "moderator":
                continue
            self.gameDescription[agent] = [
                {
                    "role": "system",
                    "content": self.real_world_setting["general"] + '\n' + self.real_world_setting[self.real_world_agents[idx]],
                }
            ]

            self.gameDescription[agent][0]['content'] = self.gameDescription[agent][0]['content'].replace("<option_A>", "J").replace("<option_B>", "F").replace("<num_rounds>", str(self.num_rounds))

        self.gameDescription["moderator"] = [
            {
                "role": "system",
                "content": f"""You are the moderator. You are moderating a conversation between two players A and B. \
Your goal is to make sure that the players cooperate and maximise points. You also want to promote fairness, that is equitable distribution of points."""
            }
        ]

        self.message_context = {agent: [] for agent in self.agents}

        self.choose_output = {
            agent: [
                {
                    "role": "user",
                    "content": f"You are player {agent}. Now choose an option between J and F. Output just a single letter J or F and nothing else.",
                }
            ]
            for agent in self.agents[:2]
        }

        self.conversation_output = {
            agent: [
                {
                    "role": "user",
                    "content": f"You are player {agent}. You will start a conversation with player {self.agents[1-i]} about your proposed strategy. Your response should atmost 2 lines.",
                }
            ]
            for i,agent in enumerate(self.agents[:2])
        }
        self.conversation_output["moderator"] = [
            {
                "role": "system",
                "content": f"You are the conversation moderator. Help the two players A and B negotiate to reach a conclusion. Your response should be atmost 2 lines."
            }
        ]

        self.points = {"A": 0, "B": 0}

    def simulate_round(self):
        for i in tqdm(range(self.num_conversation_rounds)):
            for agent in self.agents:
                if agent == "moderator" and not self.include_moderator:
                    continue
                response = llama_response(
                    "player " + agent if agent != "moderator" else "moderator",
                    self.gameDescription[agent]
                    + self.message_context[agent]
                    + self.conversation_output[agent],
                )
                
                for a in self.agents:
                    self.message_context[a] += [
                        {"role": agent, "content": response}
                    ]

    def generate_output(self):
        agent_probs = {}
        agent_actions = {}
        for agent in self.agents[:2]:
            token_probs = llama_output(
                agent, 
                self.gameDescription[agent]
                + self.message_context[agent]
                + self.choose_output[agent],
            )

            agent_probs[agent] = {action: token_probs[action] if action in token_probs.keys() else 0 for action in self.actions}
            sum_probs = sum(agent_probs[agent].values())

            agent_probs[agent] = {k: v / sum_probs for k, v in agent_probs[agent].items()}
            
            agent_actions[agent] = 'J' if agent_probs[agent]['J'] > agent_probs[agent]['F'] else 'F'

        return agent_actions, agent_probs

    def simulate_game(self):
        df = pd.DataFrame(
            columns=["round"]
            + [f"action_{agent}" for agent in self.agents[:2]]
            + [f"points_{agent}" for agent in self.agents[:2]]
            + [f"probs_{agent}" for agent in self.agents[:2]]
        )

        self.simulate_round()

        points = {"A": 0, "B": 0}

        for i in tqdm(range(self.num_rounds)):
            agent_actions, agent_probs = self.generate_output()

            mask = np.array([[agent_probs['A'][action_A] * agent_probs['B'][action_B] for action_B in self.actions] for action_A in self.actions])

            for idx, agent in enumerate(self.agents[:2]):
                self.points[agent] += np.sum(
                    mask * np.array(self.utility_matrix)[:, :, idx], axis=None
                )

                points[agent] = self.utility_matrix[agent_actions[agent] == self.actions[1]][agent_actions[self.agents[1-idx]] == self.actions[1]][idx]

            round_output = {agent: self.feedback_matrix[agent_actions[agent] == self.actions[1]][agent_actions[self.agents[1-idx]] == self.actions[1]][idx] for idx, agent in enumerate(self.agents[:2])}
            
            for agent in self.agents[:2]:
                self.message_context[agent] += [{"role": "system", "content": round_output[agent]}]

            qualitative_scores = {}
            for idx, agent in enumerate(self.agents[:2]):
                response = llama_response(
                    "player " + agent if agent != "moderator" else "moderator",
                    self.gameDescription[agent]
                    + self.message_context[agent]
                    + self.evaluation[agent],
                )
                print(response)
                try:
                    numbers = [int(num) for num in re.findall(r'>(\d+)<', response)]
                    qualitative_scores[agent] = numbers  # Initialize the list directly
                except Exception as e:
                    qualitative_scores[agent] = [5.0] * 5  # Default value in case of an error

            # Ensure both agents have scores
            if 'A' not in qualitative_scores or len(qualitative_scores['A']) != 5:
                qualitative_scores['A'] = [5.0] * 5  # Default values
            if 'B' not in qualitative_scores or len(qualitative_scores['B']) != 5:
                qualitative_scores['B'] = [5.0] * 5  # Default values

            new_row = pd.DataFrame(
                {
                    "round": [i + 1],
                    "action_A": [agent_actions["A"]],
                    "action_B": [agent_actions["B"]],
                    "points_A": [self.points["A"]],
                    "points_B": [self.points["B"]],
                    "probs_A": [agent_probs["A"]],
                    "probs_B": [agent_probs["B"]],
                    "qualitative_scores_A": [np.mean(qualitative_scores["B"])],
                    "qualitative_scores_B": [np.mean(qualitative_scores["A"])],
                    "T_A": [qualitative_scores["B"][0]],
                    "C_A": [qualitative_scores["B"][1]],
                    "CS_A": [qualitative_scores["B"][2]],
                    "R_A": [qualitative_scores["B"][3]],
                    "S_A": [qualitative_scores["B"][4]],
                    "T_B": [qualitative_scores["A"][0]],
                    "C_B": [qualitative_scores["A"][1]],
                    "CS_B": [qualitative_scores["A"][2]],
                    "R_B": [qualitative_scores["A"][3]],
                    "S_B": [qualitative_scores["A"][4]],
                }
            )

            # Concatenate the new row with the existing DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

        return df, self.message_context, self.gameDescription, self.conversation_output, self.choose_output


def simulate_and_save(real_world_game_setting, num_conversation_rounds, include_moderator, i):
    game_sim = Game(real_world_game_setting, 10, num_conversation_rounds, include_moderator)
    df, messages, gameDescription, conversation_prompt, choose_output_prompt = game_sim.simulate_game()

    dump = {
        'gameDescription': gameDescription,
        'conversation_prompt': conversation_prompt,
        'choose_output_prompt': choose_output_prompt,
        'messages': messages,
    }

    os.makedirs(f"logs_real_world/{real_world_game_setting['game']}/{real_world_game_setting['scenario']}", exist_ok=True)

    df.to_csv(
        f"logs_real_world/{real_world_game_setting['game']}/{real_world_game_setting['scenario']}/game_{num_conversation_rounds}_{include_moderator}_{i}.csv",
        index=False,
    )

    with open(
        f"logs_real_world/{real_world_game_setting['game']}/{real_world_game_setting['scenario']}/game_{num_conversation_rounds}_{include_moderator}_{i}.json", "w"
    ) as f:
        json.dump(dump, f)


if __name__ == "__main__":
    # List of parameters for each simulation
    parser = argparse.ArgumentParser(description="Run simulations for synthetic games")
    parser.add_argument("--game", type=int, help="Type of game to simulate")

    args = parser.parse_args()

    f = os.listdir('realworld2x2')

    real_world_game_settings = json.loads(open(f'realworld2x2/{f[args.game]}').read())

    print(real_world_game_settings)

    tasks = [
        (real_world_game_setting, num_conversation_rounds, include_moderator, i)
        for real_world_game_setting in real_world_game_settings
        for num_conversation_rounds in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for include_moderator in [False, True]
        for i in range(2)
    ]

    # print("All tasks completed successfully!")
    for task in tqdm(tasks):
        simulate_and_save(task[0], task[1], task[2], task[3])

