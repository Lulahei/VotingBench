from eval.eval_utils import *
# from eval.player import *
from eval.player_zh import *
import time
# time.sleep(7200)
import argparse
import copy

def run_img_distribution(args, num_per_player):
    file_path = os.path.join(args.output_path, f'all_pairs_{len(args.player_list)}_player_{num_per_player}_hands.json')
    if os.path.exists(file_path):
        return json.load(open(file_path, 'r'))
    all_pairs = get_random_pair_dict(args.image_path, len(args.player_list), num_per_player, args.output_path)

    return all_pairs

def run_description_generation(args, all_pairs, player):
    ret = copy.deepcopy(all_pairs)
    for i in range(len(all_pairs)):
        ori_img = all_pairs[i]['ori_img']
        player.update_hand([ori_img])
        inputs, _ = player.prepare_narrator_say_input(description_type=args.description_type)
        while True:
            try:
                description_raw = player.generate_response(inputs)
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
            break

        try:
            description = get_description_from_raw(description_raw)
        except Exception as e:
            print(e)
            description = str(e)
            
        ret[i]['description'] = description
    
    return ret
        

def run_img_selection(args, descriptions, player):
    imgs_with_selection = []
    for desc in descriptions:
        idx = get_player_idx(args.narrator_model, args.model, args.player_list)
        derived_hands = desc["player_hands"][idx]
        player.update_hand(derived_hands)
        if args.run_type != 'local':
            inputs = player.prepare_select_input(desc['description'])
            while True:
                try:
                    text_raw = player.generate_response(inputs)
                    # print(text_raw)
                    img_sel = get_idx_from_raw(text_raw, player.hand)
                except Exception as e:
                    print(e)
                    time.sleep(15)
                    continue
                break           
        else:
            img_sel = player.select_image_ppl(desc['description'])

        imgs_with_selection.append(dict(
            ori_img = desc['ori_img'],
            description = desc['description'],
            selected_img = img_sel
        ))
            
    return imgs_with_selection


def run_image_voting(args, descriptions, player):
    imgs_voted = []
    for desc in descriptions:
        all_voting_pics = list(desc['selected_img'].values())
        all_voting_pics.append(desc['ori_img'])
        all_voting_pics.remove(desc['selected_img'][args.model])
        random.shuffle(all_voting_pics)
        if args.run_type != 'local':
            inputs = player.prepare_vote_input(desc['description'], all_voting_pics)
            while True:
                try:
                    text_raw = player.generate_response(inputs)
                    # print(text_raw)
                except Exception as e:
                    print(e)
                    time.sleep(15)
                    continue
                break

            try:
                img_vot = get_idx_from_raw(text_raw, all_voting_pics)
            except Exception as e:
                print(e)
                img_vot = str(e) + " | " + text_raw + " | " + str(all_voting_pics)
        else:
            img_vot = player.select_image_ppl(desc['description'], all_voting_pics)

        imgs_voted.append(dict(
            ori_img = desc['ori_img'],
            guessed_img = img_vot
        ))
    
    return imgs_voted

def main(args):
    # get pics
    all_pairs = eval(f'run_img_distribution(args, 6)')

    # create player obj
    model_player = {
    "Qwen2": player4Qwen2VL,
    "GLM4V": player4GLM4V,
    "human": player4Human,
    "Qwen": player4QwenVL,
    "InternVL2": player4InternVL2,
    "LLama": player4LLama,
    "DeepSeekVL": player4DeepSeekVL,
    "Step": player4STEP1V,
    "ZeroOne": player4YiVisionV2
    }

    json_path = os.path.join(args.output_path, f'descriptions_{args.round_per_player}_imgs_{args.description_type}.json')
    # generate description
    if args.running_step == 'run_description_generation':

        player = model_player[args.model](args.model, None, args.run_type)

        descriptions = run_description_generation(args, all_pairs[:args.round_per_player], player)

        ret = {}
        # write in json
        if os.path.exists(json_path):
            ret = json.load(open(json_path, 'r', encoding='utf-8'))
            ret[f"{args.model}"] = descriptions
        else:
            ret = {f"{args.model}": descriptions}

        json.dump(ret, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # image_selection
    elif args.running_step == 'run_img_selection':
        player = model_player[args.model](args.model)

        # get select img list
        generated_descriptions = json.load(open(json_path, 'r', encoding='utf-8'))[args.narrator_model]
        selected_list = run_img_selection(args, generated_descriptions, player)

        # write in json
        write_path = os.path.join(args.output_path, f'result_{args.round_per_player}_imgs_{args.description_type}.json')
        if os.path.exists(write_path):
            ret = json.load(open(write_path, 'r', encoding='utf-8'))
            if args.narrator_model not in ret.keys():
                finished_step = []
                for i in selected_list:
                    finished_step.append(dict(
                        ori_img = i['ori_img'],
                        description = i['description'],
                        selected_img = {f"{args.model}": i['selected_img']}
                    ))
                ret[f"{args.narrator_model}"] = finished_step
            else:
                for i in selected_list:
                    idx = get_piece_idx(ret[args.narrator_model], i['ori_img'])
                    ret[args.narrator_model][idx]['selected_img'][args.model] = i['selected_img']
            
        else:
            finished_step = []
            for i in selected_list:
                finished_step.append(dict(
                    ori_img = i['ori_img'],
                    description = i['description'],
                    selected_img = {f"{args.model}": i['selected_img']}
                ))
            ret = {f"{args.narrator_model}": finished_step}

        json.dump(ret, open(write_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    # image_voting
    elif args.running_step == 'run_image_voting':
        player = model_player[args.model](args.model)

        # get select img list
        json_path = os.path.join(args.output_path, f'result_{args.round_per_player}_imgs_{args.description_type}_zh.json')
        generated_descriptions_all = json.load(open(json_path, 'r', encoding='utf-8'))
        generated_descriptions = generated_descriptions_all[args.narrator_model]
        guessed_list = run_image_voting(args, generated_descriptions, player)

        # write in json
        for i in guessed_list:
            idx = get_piece_idx(generated_descriptions, i['ori_img'])
            if 'guessed_img' not in generated_descriptions[idx].keys():
                generated_descriptions[idx]['guessed_img'] = {}
            generated_descriptions[idx]['guessed_img'][args.model] = i['guessed_img']
        
        generated_descriptions_all[args.narrator_model] = generated_descriptions
        json.dump(generated_descriptions_all, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--output_path', type=str, default="results/v3_res_human", help='output_path')
    parser.add_argument('--model', type=str, default="human", help='eval model name')
    parser.add_argument('--run_type', type=str, default="human", help='model generation type')
    parser.add_argument('--player_list', type=list, default=["Qwen2", "GLM4V", "InternVL2", "DeepSeekVL", "LLama", "Step", "human"], help='123')
    parser.add_argument('--round_per_player', type=int, default=50, help='123')
    parser.add_argument('--image_path', type=str, default="/data1/zjy/DIXIT/data/images", help='DIXIT image path')
    parser.add_argument('--running_step', type=str, default="run_image_voting", help='DIXIT current running step')
    parser.add_argument('--narrator_model', type=str, default="DeepSeekVL", help='123')
    parser.add_argument('--description_type', type=str, default="rule", help='123')
    args = parser.parse_args()  # 获取所有参数

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main(args)