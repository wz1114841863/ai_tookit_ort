import json

items = []
with open("./config.json", "r") as fp:
    data = json.load(fp)
    for key, item in data["id2label"].items():
        items.append(item)

print(items)
