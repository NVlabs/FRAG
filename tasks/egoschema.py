from .base import VideoTask
from .utils import get_multi_choice_info, parse_multi_choice_response
import requests


class EgoSchema(VideoTask):
    def __init__(self, dataset, split, **kwargs):
        super().__init__(dataset, split, **kwargs)

        assert self.split in ['subset', 'full']

    def aggregate_results(self, docs, out_root):
        if self.split == "subset":
            return super().aggregate_results(docs, out_root)
        elif self.split == "full":
            out_dict = {}
            for doc in docs:
                pred = doc["pred"][0]
                index2ans, all_choices = get_multi_choice_info(doc)
                parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
                if parsed_pred not in all_choices:
                    res = -1
                else:
                    res = all_choices.index(parsed_pred)
                out_dict[doc["id"]] = res
            
            url = "https://validation-server.onrender.com/api/upload/"
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(url, headers=headers, json=out_dict)
            
            out_file = out_root + '.log'
            with open(out_file, 'a') as file:
                file.write(response.text)
                file.write('\n')
            
