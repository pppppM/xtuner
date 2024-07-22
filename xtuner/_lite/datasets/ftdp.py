from typing import Dict, List

from transformers import PreTrainedTokenizer

from xtuner._lite import get_logger
from xtuner.utils import IGNORE_INDEX

logger = get_logger()

ROLE_CFG = dict(
    knowledge=dict(
        begin=dict(without_name='<|im_start|>knowledge\n', ),
        end='<|im_end|>\n',
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        )),
    system=dict(
        begin=dict(
            with_name='<|im_start|>system name={name}\n',
            without_name='<|im_start|>system\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n',
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        )),
    user=dict(
        begin=dict(
            with_name='<|im_start|>user name={name}\n',
            without_name='<|im_start|>user\n',
        ),
        end='<|im_end|>\n',
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        )),
    assistant=dict(
        begin=dict(
            with_name='<|im_start|>assistant name={name}\n',
            without_name='<|im_start|>assistant\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n',
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        )),
    environment=dict(
        begin=dict(
            with_name='<|im_start|>environment name={name}\n',
            without_name='<|im_start|>environment\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|im_end|>\n',
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        )),
    tool=dict(
        begin=dict(
            with_name='<|action_start|>{name}\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            }),
        end='<|action_end|>\n',
        belong='assistant',
    ),
    thought=dict(
        begin=dict(without_name=''),
        end='',
        belong='assistant',
    ),
)

MAX_LEN = 32 * 1024


def format_begin(role_cfg, message):
    name = message.get('name', None)
    if name is not None:
        begin = role_cfg['begin'].get('with_name', '')
        if name in role_cfg['begin'].get('name', {}):
            begin = begin.format(name=role_cfg['begin']['name'][name])
        else:
            begin = begin.format(name=name)
    else:
        begin = role_cfg['begin'].get('without_name', '')
    return begin


def format_sub_role(messages: List[Dict], roles_cfg) -> List[Dict]:
    new_message = list()
    for message in messages:
        if message['role'] in ['assistant', 'user', 'system', 'environment']:
            new_message.append(message)
            continue
        role_cfg = roles_cfg[message['role']]
        begin = format_begin(role_cfg, message)
        new_content = begin + message['content'] + role_cfg['end']
        if role_cfg.get('fallback_role'):
            new_message.append(
                dict(role=role_cfg['fallback_role'], content=new_content))
        elif role_cfg.get('belong'):
            if new_message[-1]['role'] != role_cfg.get('belong'):
                new_message.append(
                    dict(role=role_cfg.get('belong'), content=new_content))
            else:
                new_message[-1]['content'] += new_content
        else:
            new_message.append(dict(role=message['role'], content=new_content))

    return new_message


def ftdp_tokenize(tokenizer: PreTrainedTokenizer, messages) -> Dict:
    token_ids = []
    _processed_data = format_sub_role(messages, ROLE_CFG)

    for dialog_item in _processed_data:
        role = dialog_item['role']
        content = dialog_item['content']
        # TODO: is strip necessary? or use lstrip? 避免开始有\n\n的情况
        # content = content.lstrip()
        begin = format_begin(ROLE_CFG[role], dialog_item)
        end = ROLE_CFG[role]['end']
        begin_token = tokenizer.encode(begin, add_special_tokens=False)
        if not ROLE_CFG[role]['loss'].get('beigin', False):
            begin_token = [-token_id for token_id in begin_token]
        end_token = tokenizer.encode(
            ROLE_CFG[role]['end'], add_special_tokens=False)
        # breakpoint()
        if not ROLE_CFG[role]['loss'].get('end', False):
            end_token = [-token_id for token_id in end_token]

        content_token = tokenizer.encode(
            begin + content + end, add_special_tokens=False)
        content_token = content_token[len(begin_token):-len(end_token)]

        if dialog_item.get('loss', True):
            loss_cfg = ROLE_CFG[role]['loss']
        else:
            loss_cfg = dict(icl=False, current=False, meta=False)

        loss_type = dialog_item.get('type', 'current')
        if (loss_type in loss_cfg) and (not loss_cfg[loss_type]):
            content_token = [-token_id for token_id in content_token]

        if begin == '':
            tokens = content_token
        else:
            tokens = begin_token + content_token
        if end != '':
            tokens = tokens + end_token

        token_ids += tokens

    token_ids = [tokenizer.bos_token_id] + token_ids
    token_ids = token_ids[:MAX_LEN]
    labels = [x if x > 0 else IGNORE_INDEX for x in token_ids]
    token_ids = [abs(x) for x in token_ids]

    training_data = {
        'input_ids': token_ids,
        'labels': labels,
        'num_tokens': len(token_ids),
    }

    return training_data


class FtdpTokenizeFunction():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, item):
        return ftdp_tokenize(self.tokenizer, item)


def map_ftdp_tokenized_data(item):
    item['input_ids'] = item['tokens']
    del item['tokens']
    labels = [x if x > 0 else -100 for x in item['input_ids']]
    item['input_ids'] = [abs(x) for x in item['input_ids']]
    item['labels'] = labels
    item['num_tokens'] = len(item['input_ids'])
    return item
