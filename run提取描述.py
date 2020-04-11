from config import *
Ner_Marker=Marker(model_path="./tkitfiles/tmarker_bert_ner")
Ner_Marker.load_model()




for it in DB.content_pet.find():
    tt= tkitText.Text()
    c_id=tt.md5(it['content'])
    if DB.entity_kg_content.find_one({"_id":c_id})==None:
        print("###"*10)
        print(it['title'])
        pall=Ner_Marker.pre_ner(it['title']+"。"+it['content'])
        # print(pall)
        for one_a in pall:
            one=Pred_Marker.pre(one_a,it['content'])
            # print(one)
            for kg in one:
                print(one_a,kg)
                add_miaoshu(one_a,kg,it['content'])
