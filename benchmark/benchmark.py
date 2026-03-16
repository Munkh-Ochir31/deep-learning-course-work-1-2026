import json
import random

dataset = []
id_counter = 1

cities = ["Дархан","Эрдэнэт","Ховд","Чойбалсан"]
math_ops = ["+", "-", "*"]

# ---------- MCQ ----------
for i in range(200):

    choices = cities + ["Улаанбаатар"]
    random.shuffle(choices)

    dataset.append({
        "id": id_counter,
        "task": "mcq",
        "question": "Монгол улсын нийслэл аль хот вэ?",
        "choices": choices,
        "answer": "Улаанбаатар",
        "difficulty": "easy"
    })

    id_counter += 1


# ---------- MATH ----------
for i in range(150):

    a = random.randint(1,20)
    b = random.randint(1,20)
    op = random.choice(math_ops)

    if op == "+":
        ans = a+b
    elif op == "-":
        ans = a-b
    else:
        ans = a*b

    dataset.append({
        "id": id_counter,
        "task": "math",
        "question": f"{a} {op} {b} хэд вэ?",
        "answer": str(ans),
        "difficulty": "easy"
    })

    id_counter += 1


# ---------- READING ----------
contexts = [
"Монгол улс 1924 онд БНУ болсон.",
"Чингис хаан 1206 онд Монголын эзэнт гүрнийг байгуулсан.",
"Говь цөл Монголын өмнөд хэсэгт оршдог."
]

for i in range(200):

    ctx = random.choice(contexts)

    if "1924" in ctx:
        q="Монгол улс хэдэн онд БНУ болсон бэ?"
        a="1924"

    elif "1206" in ctx:
        q="Монголын эзэнт гүрэн хэдэн онд байгуулагдсан бэ?"
        a="1206"

    else:
        q="Говь цөл Монголын аль хэсэгт байдаг вэ?"
        a="өмнөд хэсэгт"

    dataset.append({
        "id": id_counter,
        "task": "reading",
        "context": ctx,
        "question": q,
        "answer": a,
        "difficulty": "easy"
    })

    id_counter += 1


# ---------- COMMONSENSE ----------

commonsense = [
("Бороо орж байвал хүмүүс юу авч явдаг вэ?","шүхэр"),
("Хүн өлссөн үедээ юу хийдэг вэ?","хоол иддэг"),
("Хүн их ядарсан бол юу хийдэг вэ?","амардаг")
]

for i in range(150):

    q,a=random.choice(commonsense)

    dataset.append({
        "id": id_counter,
        "task": "commonsense",
        "question": q,
        "answer": a,
        "difficulty": "easy"
    })

    id_counter+=1


# ---------- TRANSLATION ----------

translations=[
("Монгол улс бол Азийн орон","Mongolia is a country in Asia"),
("Тэнгэр цэнхэр","The sky is blue"),
("Би ном уншиж байна","I am reading a book")
]

for i in range(150):

    mn,en=random.choice(translations)

    dataset.append({
        "id": id_counter,
        "task": "translation",
        "question": f"Дараах өгүүлбэрийг англи хэл рүү орчуул: {mn}",
        "answer": en,
        "difficulty": "medium"
    })

    id_counter+=1


# ---------- INSTRUCTION ----------

texts=[
"Монгол улс нь Зүүн Азид орших ардчилсан улс юм.",
"Улаанбаатар бол Монголын хамгийн том хот."
]

for i in range(100):

    t=random.choice(texts)

    dataset.append({
        "id": id_counter,
        "task": "instruction",
        "question": f"Дараах өгүүлбэрийг товчло: {t}",
        "answer": t,
        "difficulty": "medium"
    })

    id_counter+=1


# ---------- SPELLING ----------

spelling=[
("монгол улсаа нийслэл","Монгол улсын нийслэл"),
("би ном уншж бна","би ном уншиж байна")
]

for i in range(50):

    wrong,correct=random.choice(spelling)

    dataset.append({
        "id": id_counter,
        "task": "grammar",
        "question": f"Дараах өгүүлбэрийн алдааг зас: {wrong}",
        "answer": correct,
        "difficulty": "medium"
    })

    id_counter+=1


# save
with open("mongolian_llm_benchmark.jsonl","w",encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item,ensure_ascii=False)+"\n")

print("Dataset size:",len(dataset))