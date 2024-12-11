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
)

def llama_response(role, messages):
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False) + f"<|start_header_id|>{role}<|end_header_id|>\n"
    outputs = base_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.5,
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
        self, utility_matrix, num_rounds, num_conversation_rounds, include_moderator
    ):

        self.num_rounds = num_rounds
        self.num_conversation_rounds = num_conversation_rounds
        self.include_moderator = include_moderator

        self.actions = ["J", "F"]

        # self.utility_matrix = [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]
        self.utility_matrix = utility_matrix

        self.agents = ["A", "B", "moderator"]

        self.gameDescription = {agent: "" for agent in self.agents}

        for idx, agent in enumerate(self.agents):
            if agent == "moderator":
                continue
            self.gameDescription[agent] = [
                {
                    "role": "system",
                    "content": f"""Imagine you are {agent}, playing a game repeatedly with another \
player {self.agents[1-idx]}. In this game, you can choose between Option J and Option F. You will play this game for {self.num_rounds} rounds in total \
with the same player. Before picking options, you can communicate with the other player for {self.num_conversation_rounds} conversation rounds. \
After you converse with the other player, you have to choose an Option. You can trick the other player to maximize your points. \
You cannot trust the other player to say the truth always. \
{"A moderator will help you and other player to reach optimal strategy. " if self.include_moderator else ""}\
The rules of the game are as follows: \
If you choose Option J and the other player chooses \
Option J, then you get {self.utility_matrix[0][0][idx]} points and the other \
player wins {self.utility_matrix[0][0][1-idx]} points. \
If you choose Option J and the other player chooses \
Option F, then you get {self.utility_matrix[0][1][idx]} points and the other player \
wins {self.utility_matrix[0][1][1-idx]} points. \
If you choose Option F and the other player chooses \
Option J, then you get {self.utility_matrix[1][0][idx]} points and the other player \
wins {self.utility_matrix[1][0][1-idx]} points. \
If you choose Option F and the other player chooses \
Option F, then you get {self.utility_matrix[1][1][idx]} points and the other player \
wins {self.utility_matrix[1][1][1-idx]} points. \
Your goal is to maximize your total points over {self.num_rounds} rounds.""",
                }
            ]

        self.gameDescription["moderator"] = [
            {
                "role": "system",
                "content": f"""You are the moderator. You are moderating a conversation between two players A and B. \
Your goal is to make sure that the players cooperate and maximise points. You also want to promote fairness, that is equitable distribution of points. \
The rules of the game are as follows: \
If player A choose Option J and player B chooses \
Option J, then A gets {self.utility_matrix[0][0][0]} points and B gets {self.utility_matrix[0][0][1]} points. \
If player A choose Option J and player B chooses \
Option F, then A gets {self.utility_matrix[0][1][0]} points and B gets {self.utility_matrix[0][1][1]} points. \
If player A choose Option F and player B chooses \
Option J, then A gets {self.utility_matrix[1][0][0]} points and B gets {self.utility_matrix[1][0][1]} points. \
If player A choose Option F and player B chooses \
Option F, then A gets {self.utility_matrix[1][1][0]} points and B gets {self.utility_matrix[1][1][1]} points.""",
            }
        ]

        self.message_context = []

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
        
        self.conversation_output["moderator"] = [
            {
                "role": "user",
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
                    + self.message_context
                    + self.conversation_output[agent],
                )

                self.message_context += [
                    {"role": agent, "content": response}
                ]

    def generate_output(self):
        agent_probs = {}
        agent_actions = {}

        for agent in self.agents[:2]:
            token_probs = llama_output(
                agent, 
                self.gameDescription[agent]
                + self.message_context
                + self.choose_output[agent],
            )

            agent_probs[agent] = {action: token_probs[action] if action in token_probs.keys() else 0 for action in self.actions}
            sum_probs = sum(agent_probs[agent].values())

            if sum_probs == 0:
                print("SOMETHING WENT WRONG")

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

            round_output = f"""In this round, Player {self.agents[0]} chose {agent_actions['A']} and Player {self.agents[1]} chose {agent_actions['B']}. \
Player A gets {points['A']} points and Player B gets {points['B']} points."""

            self.message_context += [{"role": "system", "content": round_output}]
            
            qualitative_scores = {}
            for idx, agent in enumerate(self.agents[:2]):
                response = llama_response(
                    "player " + agent if agent != "moderator" else "moderator",
                    self.gameDescription[agent]
                    + self.message_context
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
    
    def simulate_output_rounds(self):
        for i in range(self.num_rounds):
            agent_actions, agent_probs = self.generate_output()

            mask = np.array([[agent_probs['A'][action_A] * agent_probs['B'][action_B] for action_B in self.actions] for action_A in self.actions])

            for idx, agent in enumerate(self.agents[:2]):
                self.points[agent] += np.sum(
                    mask * np.array(self.utility_matrix)[:, :, idx], axis=None
                )

        return min(self.points['A'], self.points['B'])
            


def simulate_and_save(game_setting, num_conversation_rounds, include_moderator, i):
    game, utility_matrix = game_setting
    game_sim = Game(utility_matrix, 10, num_conversation_rounds, include_moderator)
    df, messages, gameDescription, conversation_prompt, choose_output_prompt = game_sim.simulate_game()

    dump = {
        'gameDescription': gameDescription,
        'conversation_prompt': conversation_prompt,
        'choose_output_prompt': choose_output_prompt,
        'messages': messages,
    }

    os.makedirs(f"logs_synthetic_qualitative/{game}", exist_ok=True)

    df.to_csv(
        f"logs_synthetic_qualitative/{game}/game_{num_conversation_rounds}_{include_moderator}_{i}.csv",
        index=False,
    )

    with open(
        f"logs_synthetic_qualitative/{game}/game_{num_conversation_rounds}_{include_moderator}_{i}.json", "w"
    ) as f:
        json.dump(dump, f)




if __name__ == "__main__":
    # List of parameters for each simulation
    parser = argparse.ArgumentParser(description="Run simulations for synthetic games")
    parser.add_argument("--game", default=0, type=int, help="Type of game to simulate")

    args = parser.parse_args()

    game_settings = json.loads(open("/simurgh/u/akhatua/moderation_protocol/game_settings.json").read())

    games = []
    games_to_utility = {}

    for game in game_settings:
        games.append(game["game"])
        games_to_utility[game["game"]] = game["utility"]

    tasks = [
        ((game, games_to_utility[game]), num_conversation_rounds, include_moderator, i)
        for game in [games[args.game]]
        for num_conversation_rounds in [2, 3]
        for include_moderator in [False, True]
        for i in range(2)
    ]

    # Number of processes to use
    # num_processes = multiprocessing.cpu_count()

    # print("Number of processes:", num_processes)

    # # Run tasks in parallel using Pool
    # with multiprocessing.Pool(num_processes) as pool:
    #     for _ in pool.starmap(simulate_and_save, tasks):
    #         pass

    # print("All tasks completed successfully!")
    for task in tqdm(tasks):
        simulate_and_save(task[0], task[1], task[2], task[3])