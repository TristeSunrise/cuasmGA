
from copy import deepcopy
from functools import lru_cache

import numpy as np
def decode(line: str):
    line = line.strip('\n')
    line = line.split(' ')
    n = len(line)

    ctrl_code = None
    predicate = None
    comment = None
    opcode = None
    dest = None
    src = []
    meta = {
        'reuse': [],
    }

    # ctrl
    idx = -1
    for i in range(0, n):
        if line[i] != '':
            idx = i
            ctrl_code = line[i]
            break
    assert idx > -1, f'no ctrl: {line}'

    if ctrl_code.startswith('.'):
        # labels
        return None, None, None, None, None, None, None

    # comment
    for i in range(idx + 1, n):
        if line[i] != '':
            idx = i
            comment = line[i]
            break

    # predicate
    for i in range(idx + 1, n):
        if line[i] != '':

            if line[i][0] == '@':
                predicate = line[i]
            else:
                opcode = line[i]

            idx = i
            break

    # opcode
    if opcode is None:
        for i in range(idx + 1, n):
            if line[i] != '':
                opcode = line[i]
                idx = i
                break

    # operand
    for i in range(idx + 1, n):
        if line[i] != '':
            dest = line[i].strip(',')
            idx = i
            break

    if dest == ';':
        # LDGDEPBAR inst
        dest = None

    for i in range(idx + 1, n):
        if line[i] == ';':
            break

        if line[i] != '':
            src.append(line[i].strip(','))

    # post-process src; e.g. ['desc[UR16][R10.64] -> UR16, R10
    processed_src = []
    for i, word in enumerate(src):
        if word.startswith('desc'):
            w = word.replace(']', '').split('[')
            for r in w[1:]:
                strip_plus = r.split('+')[0]  # R10.64+0x80 -> R10.64
                strip_64 = strip_plus.split('.')[0]  # R10.64 -> R10
                processed_src.append(strip_64)

                # hidden deps
                if strip_plus.endswith('.64'):
                    val = int(strip_64[1:])
                    base = val // 2
                    mod = val % 2
                    comp = 1 - mod
                    hidden = base * 2 + comp
                    processed_src.append(f'R{hidden}')

        elif word.startswith('c'):
            processed_src.append(word)
        else:
            tmp = word.strip(']').strip('[')
            tmp = tmp.split('+')[0]  # R10+0x2000 -> R10

            # XXX some possible suffix
            # [R153.X4+0x10]
            # SR_CTAID.Y
            # 1.4426950216293334961
            # R0.reuse
            # if len(tmp.split('.')) > 1:
            #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            #     print(word)
            #     print(tmp)
            #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            # XXX

            if tmp.endswith('reuse'):
                meta['reuse'].append(tmp)

            tmp = tmp.split('.')[0]  # R10.64 -> R10
            processed_src.append(tmp)

    # predicate should be considered as src
    if predicate is not None:
        tmp = predicate[1:]
        if tmp[0] == '!':
            tmp = tmp[1:]
        processed_src.append(tmp)

    # post-process dest; e.g. [R219+0x4000] -> R219
    if dest is not None:
        if dest.startswith('desc'):
            w = dest.replace(']', '').split('[')
            for r in w[1:]:
                strip_plus = r.split('+')[0]  # R10.64+0x80 -> R10.64
                strip_64 = strip_plus.split('.')[0]  # R10.64 -> R10
                # In this case, it is treated as src
                # e.g. STG.E desc[UR16][R10.64], R197 ;
                # the dst needs to be ready
                processed_src.append(strip_64)

                # hidden deps
                if strip_plus.endswith('.64'):
                    val = int(strip_64[1:])
                    base = val // 2
                    mod = val % 2
                    comp = 1 - mod
                    hidden = base * 2 + comp
                    processed_src.append(f'R{hidden}')
        else:
            dest = dest.strip(']').strip('[')
            dest = dest.split('.')[0]
            dest = dest.split('+')[0]

    # a hack for internal label
    if ctrl_code.startswith('$__'):
        ctrl_code = None
    return ctrl_code, comment, predicate, opcode, dest, processed_src, meta

def decode_ctrl_code(ctrl_code: str):
    ctrl_code = ctrl_code.split(':')
    assert len(ctrl_code) == 5, f'invalid ctrl code: {ctrl_code}'

    barr = ctrl_code[0][2:]
    waits = []
    for bar in barr:
        if bar != '-':
            waits.append(int(bar))

    read = ctrl_code[1]
    write = ctrl_code[2]
    yield_flag = ctrl_code[3]
    stall_count = ctrl_code[4]
    return waits, read, write, yield_flag, stall_count