import argparse
import sys
from llms import *

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--system', default=False, action='store_true')
    parser.add_argument('--demo', default=False, action='store_true')
    parser.add_argument('--limit', default=False, action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    processor, model = load_peft_model_vision("meta-llama/Llama-3.2-90B-Vision-Instruct")

    if args.demo:
        # Demo 
        train, val, test = get_dataset(args.path)
        dev_claim = ClaimVerificationDataset(val)
        sample = dev_claim[0]
        sample_prompt = make_prompt(sample['text_evidence'][0])
        print(sample['image_evidence'][0])
        output = do_inference_vision(model, processor, sample_prompt, sample['image_evidence'][0])
        print(output)
        sys.exit()

    if args.system:
        print("Running on system evidence")
        with open("./sample_dump/pred_retrieval_dev.json", "r") as f:
            dev_claim_system = json.load(f)
        f.close()

        with open("./sample_dump/pred_retrieval_test.json", "r") as f:
            test_claim_system = json.load(f)
        f.close()
        
        if args.limit:
            print("Run with limit: {}-{}".format(args.start, args.end))
            test_claim_system = test_claim_system[args.start:args.end]
            dev_claim_system = dev_claim_system[args.start:args.end]

        if not args.test:
            print("Dev")
            result_dev = create_align_form_system(dev_claim_system, model, processor, args.path)
            with open('./mocheg_claim_llama3.2_dev_system.json' if not args.limit else './mocheg_claim_llama3.2_dev_system_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
                json.dump(result_dev, f, ensure_ascii=False, indent=4)
            f.close()
        else:
            print("Test")
            result_test = create_align_form_system(test_claim_system, model, processor, args.path)
            with open('./mocheg_claim_llama3.2_test_system.json' if not args.limit else './mocheg_claim_llama3.2_test_system_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
                json.dump(result_test, f, ensure_ascii=False, indent=4)
            f.close()
    else:
        print("Running on gold evidence")
        train, val, test = get_dataset(args.path)
        dev_claim = ClaimVerificationDataset(val)
        test_claim = ClaimVerificationDataset(test)

        if not args.test:
            print("Dev")
            result_dev = create_align_form(dev_claim, model, processor, args.path)
            with open('./mocheg_claim_llama3.2_dev.json' if not args.limit else './mocheg_claim_llama3.2_dev_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
                json.dump(result_dev, f, ensure_ascii=False, indent=4)
            f.close()
        else:
            print("Test")
            result_test = create_align_form(test_claim, model, processor, args.path)
            with open('./mocheg_claim_llama3.2_test.json' if not args.limit else './mocheg_claim_llama3.2_test_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
                json.dump(result_test, f, ensure_ascii=False, indent=4)
            f.close()