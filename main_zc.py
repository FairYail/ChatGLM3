import pandas as pd
from base_log import llog
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()


# 解析csv
def csvParse(path: str):
    # 读取指定工作表和列的 Excel 文件
    df = pd.read_csv(path)
    # 处理可能的 NaN 值
    df = df.fillna('')

    # 存储每一行的数据
    all_rows_data = df.values.tolist()
    csv_data = []
    for row in all_rows_data:
        csv_data.append(row)
    return csv_data


def parse_answer(text):
    response, history = model.chat(tokenizer, text, history=[])
    llog.info(f'{text},结论{response}')


if __name__ == '__main__':
    dataList = csvParse("./zc.csv")
    csv_data = []
    isFirst = True
    for data in dataList:
        # 第一行
        if isFirst:
            single_val = []
            for val in data:
                single_val.append(val)
            single_val.append("GLM结果")
            isFirst = False
            continue

        # 第二行开始
        single_val = []
        for val in data:
            single_val.append(val)
        parse_answer(single_val[0])
