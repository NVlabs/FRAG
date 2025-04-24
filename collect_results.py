import argparse
import json
import os


def main(args):
    docs = json.load(open(args.doc_path, "r"))
    output_path = args.result_path + '.json'
    if not os.path.exists(output_path):
        out_docs = []
        for i, doc in enumerate(docs):
            if 'id' in doc:
                doc_id = doc['id']
            else:
                doc_id = "%06d" % i
            res_path = os.path.join(args.result_path, "%s.json" % doc_id)
            try:
                doc = json.load(open(res_path, "r"))
            except Exception as error:
                print(res_path)
                raise error
            out_docs.append(doc)

        with open(output_path, "w") as f:
            json.dump(out_docs, f)
    else:
        out_docs = json.load(open(output_path, 'r'))

    if args.eval:
        from tasks.builder import build_task
        task = build_task(args.dataset, args.split)
        task.aggregate_results(out_docs, args.result_path)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--doc-path", type=str, required=True)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    main(args)
