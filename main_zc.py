import pandas as pd
from base_log import llog
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
# model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()


# 解析csv
def csvParse(path: str):
    # 读取指定工作表和列的 Excel 文件
    df = pd.read_csv(path).fillna('')

    # 存储每一行的数据
    all_rows_data = df.values.tolist()
    csv_data = [["文本", "来源", "uId", "结果"]]
    for row in all_rows_data:
        csv_data.append(row)
    return csv_data


def parse_answer(text) -> str:
    text = f'你是游戏公司客服,下面是用户发出来的一些聊天和自定义昵称。语句为："{text}"。这句话是否为侮辱、辱骂言语？\nA.是侮辱 B.不是侮辱'
    response, history = model.chat(tokenizer, text,
                                   max_length=2048,
                                   top_p=0.8,
                                   temperature=0.6,
                                   history=[])
    llog.info(f'{text}\n结论{response}')
    return response


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
            csv_data.append(single_val)
            continue

        # 第二行开始
        single_val = []
        for val in data:
            single_val.append(val)
        result = parse_answer(single_val[0])
        # 判断result是否包含英文字母A
        if result.find("A") != -1:
            single_val.append("通过")
        elif result.find("B") != -1:
            single_val.append("不通过")
        else:
            single_val.append("未知")
        single_val.append(result)
        csv_data.append(single_val)

    # 写入csv
    csv_df = pd.DataFrame(csv_data)
    # 保存为CSV文件
    csv_filename = 'zc_glm.csv'  # 文件名
    csv_df.to_csv(csv_filename, index=False)  # index=False 表示不保存行索引
