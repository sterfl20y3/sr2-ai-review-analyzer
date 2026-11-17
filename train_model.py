"""Train a tiny educational sentiment model and export it for browser inference.
Run this script in Google Colab, or locally after installing scikit-learn.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

POSITIVE = [
    "Дуже хороший товар, все сподобалося",
    "Чудова покупка, я задоволений",
    "Швидка доставка і гарна якість",
    "Прекрасне обслуговування, рекомендую",
    "Все працює відмінно, дякую",
    "Товар перевершив очікування",
    "Приємна ціна та чудова якість",
    "Я дуже задоволена результатом",
    "Все супер, замовлю ще раз",
    "Чудовий сервіс та ввічливий персонал",
    "Добрий продукт, рекомендую друзям",
    "Працює добре, недоліків не знайшов",
    "Замовлення приїхало швидко, усе ціле",
    "Гарне пакування та відмінний товар",
    "Зручний сайт, усе легко знайти",
    "Класна якість, мені сподобалось",
    "Дякую за швидку та якісну роботу",
    "Дуже зручний і корисний продукт",
    "Усе чудово, залишився задоволений",
    "Відмінна річ за свою ціну",
    "Сервіс порадував, працівники допомогли",
    "Найкраща покупка цього місяця",
    "Хороша доставка, прийшло вчасно",
    "Сподобалася якість і зовнішній вигляд",
    "Дуже гарний товар, рекомендую всім",
    "Приємно користуватися, все чудово",
    "Якісний товар і швидке оформлення",
    "Результат хороший, усе працює",
    "Дуже сподобалося, буду купувати знову",
    "Чудова робота команди, велике дякую",
]

NEGATIVE = [
    "Дуже поганий товар, не сподобалося",
    "Жахлива покупка, я розчарований",
    "Повільна доставка і низька якість",
    "Погане обслуговування, не рекомендую",
    "Нічого не працює, марно витрачені гроші",
    "Товар не виправдав очікувань",
    "Завищена ціна та погана якість",
    "Я дуже незадоволена результатом",
    "Все жахливо, більше не замовлю",
    "Грубий персонал і поганий сервіс",
    "Неякісний продукт, не раджу друзям",
    "Працює погано, багато недоліків",
    "Замовлення запізнилося, упаковка пошкоджена",
    "Жахливе пакування та зламаний товар",
    "Незручний сайт, неможливо нічого знайти",
    "Низька якість, мені не сподобалось",
    "Дуже повільна та неякісна робота",
    "Незручний і непотрібний продукт",
    "Усе погано, залишився незадоволений",
    "Невдала річ за таку ціну",
    "Сервіс розчарував, працівники не допомогли",
    "Найгірша покупка цього місяця",
    "Доставка погана, прийшло із запізненням",
    "Не сподобалася якість і зовнішній вигляд",
    "Дуже поганий товар, нікому не рекомендую",
    "Неприємно користуватися, все зламалося",
    "Неякісний товар і повільне оформлення",
    "Результат поганий, нічого не працює",
    "Зовсім не сподобалося, більше не купуватиму",
    "Жахлива робота команди, даремно витратив час",
]


# Small synthetic examples reinforce obvious sentiment words.
# This is an educational dataset, not a representative review corpus.
POSITIVE_WORDS = [
    "чудовий", "хороший", "якісний", "прекрасний", "відмінний",
    "зручний", "корисний", "надійний", "класний", "приємний",
    "чудова", "гарна", "супер"
]
NEGATIVE_WORDS = [
    "поганий", "жахливий", "неякісний", "незручний", "непотрібний",
    "зламаний", "зіпсований", "невдалий", "неприємний", "огидний",
    "погана", "жахлива", "розчарування"
]
for word in POSITIVE_WORDS:
    POSITIVE.extend([f"Дуже {word} товар", f"Мій відгук про покупку: {word}", f"Сервіс і якість: {word}"])
for word in NEGATIVE_WORDS:
    NEGATIVE.extend([f"Дуже {word} товар", f"Мій відгук про покупку: {word}", f"Сервіс і якість: {word}"])

texts = POSITIVE + NEGATIVE
labels = ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE)

# First evaluate on held-out examples.
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
train_matrix = vectorizer.fit_transform(train_texts)
evaluation_model = MultinomialNB(alpha=1.0).fit(train_matrix, train_labels)
predictions = evaluation_model.predict(vectorizer.transform(test_texts))
print(f"Test accuracy on held-out examples: {accuracy_score(test_labels, predictions):.1%}")

# Refit on the full small example dataset for the educational website.
vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
full_matrix = vectorizer.fit_transform(texts)
model = MultinomialNB(alpha=1.0).fit(full_matrix, labels)

export = {
    "model_type": "MultinomialNB",
    "language": "uk",
    "classes": model.classes_.tolist(),
    "vocabulary": vectorizer.vocabulary_,
    "class_log_prior": model.class_log_prior_.tolist(),
    "feature_log_prob": model.feature_log_prob_.tolist(),
    "training_examples": len(texts),
}

output = Path("model.json")
output.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {output} ({output.stat().st_size} bytes)")
print("Note: This is a small synthetic demonstration model, not a production classifier.")
