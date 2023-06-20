"""from transformers import BertTokenizer


data_format['Tokenization_2'] = data_format['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][0], x[i][1], 10) for i in range(len(x))])

print("on enregistre dans un json le dataframe")
data_format.to_json('/home/daril_kw/data/data_2_without_time_info.json',orient='records')

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


list_token = []
for i in range(len(data_format)):
    for j in range(len(data_format['Tokenization_2'][i])):
        list_token.append(data_format['Tokenization_2'][i][j])


print("token de Tokenization ajoutes")




list_token = list(set(list_token))
print("longueur ajoutée",len(list_token))

tokenizer.add_tokens(list_token)

print("tokenizer maj 1")



tokenizer.save_pretrained('/home/daril_kw/data/tokenizer_2_pas_tout')


"""

len_context_info =6


