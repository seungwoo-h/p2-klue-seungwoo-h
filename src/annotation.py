import pandas as pd
import os
import argparse
from tqdm import tqdm

def main(START, END, PATH):

    ### preprocess to our format

    os.system('clear')

    df = pd.read_csv(PATH, sep='\t', header=None)
    df = df.iloc[START:END + 1, :]

    df_processed = pd.DataFrame(columns=['id', 'sentence', 'entity_01', 'entity_01_start', 'entity_01_end', 'entity_02', 'entity_02_start', 'entity_02_end'])

    for idx in range(len(df)):
        row = df.iloc[idx, :]
        sbj, obj, s = row[0], row[1], row[3]
        s = s.replace('_sbj_', sbj).replace('_obj_', obj)
        s = s.replace('[[', '').replace(']]', '').replace('《', '').replace('》', '')
        s = s.replace('  ', '').strip()
        sbj_start, obj_start = s.find(sbj), s.find(obj)
        sbj_end, obj_end = sbj_start + len(sbj) - 1, obj_start + len(obj) - 1
        s_id = f'gold-standard-v1-{idx+236}'
        df_processed.loc[idx] = [s_id, s, sbj, sbj_start, sbj_end, obj, obj_start, obj_end]

    ### annotation tool

    label_dct = {'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41} 

    targets = []
    for idx in tqdm(range(len(df_processed))):
        # print(label_dct, '\n', '\n')
        p = []
        for k, v in label_dct.items():
            p.append(f'{k} - {v}')
            if len(p) == 5:
                print(p)
                p = []
        row = df_processed.iloc[idx]
        print('\n')
        print(row['sentence'], '\n')
        print(row['entity_01'], '\t' , row['entity_02'], '\t - ', df.iloc[idx, 2])
        print('\n')
        while True:
            try:
                target = input('Target? - ')
                if target == '':
                    target = 0
                target = int(target)
                break
            except:
                if target == 'break': # esc
                    break
                continue
        if target == 'break':
            break
        targets.append(target)
        os.system('clear')

    df_processed['target'] = targets
    df_processed.to_csv('./new_.tsv', header=None, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=236)
    parser.add_argument("--end", type=int, default=295)
    parser.add_argument("--path", type=str, default='./agreement_content.tsv')
    args = parser.parse_args()
    main(args.start, args.end, args.path)