import pandas as pd


def parse_data(name, s_no):
    try:
        f = open("output/{}/{}.txt".format(name, s_no), "r", encoding="utf8")
        # print(f.read())
        l = []
        total_len = 0
        for x in f:
            x = x.strip()
            total_len = total_len + len(x)
            l.append(x)
        print(len(l), total_len/len(l))
    except Exception as e:
        print(e)
    return int(total_len/len(l)), len(l)


def remove_unwanted_data(name, s_no, abp):
    # try:
    f = open("output/{}/{}.txt".format(name, s_no), "r", encoding="utf8")
    # print(f.read())
    binary_file = open("processed/{}/{}.txt".format(name, s_no), "a")
    # data = ''
    total_len = 0
    count = 0
    for x in f:
        x = x.strip()
        if len(x) > abp:
            x = x  + '\n'
            try:
                binary_file.write(x)
                total_len = total_len + len(x)
                count = count + 1
            except Exception as e:
                print(e)
    f.close()
    # bytedata = data.encode('utf-8')
    binary_file.close()
    if count == 0:
        print(name, s_no, count, 0)
        return total_len, count
    else:
        print(name, s_no, count, total_len/count)
        return int(total_len/count), count
    # except Exception as e:
    #     print(e)
    
# def processing():
#     df = pd.read_csv('new_data.csv')
#     for i in range(len(df)):
#         s_no, name = df['s_no'][i], df['name'][i]
#         average_before_processing, total_lines = parse_data(name, s_no)
#         # average_after_processing = average_after_processing - (average_after_processing % 10)
#         df['abp'][i] = average_before_processing
#         df['lbp'][i] = total_lines
#     if 'Unnamed: 0' in df.columns:
#         df = df.drop(columns=['Unnamed: 0'])
#     print(df.columns)
#     column_list = ["s_no","url","name","abp","lbp","aap","lap"]
#     df.to_csv('new_data.csv', columns = column_list)


def processing():
    df = pd.read_csv('new_data.csv')
    for i in range(len(df)):
        s_no, name, abp = df['s_no'][i], df['name'][i], df['abp'][i]
        average_after_processing, total_lines = remove_unwanted_data(name, s_no, abp)
        average_after_processing = average_after_processing - (average_after_processing % 10)
        df['aap'][i] = average_after_processing
        df['lap'][i] = total_lines
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    print(df.columns)
    column_list = ["s_no","url","name","abp","lbp","aap","lap"]
    df.to_csv('new_data.csv', columns = column_list)

processing()