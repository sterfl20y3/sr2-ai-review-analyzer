# AI Review Analyzer

**Тема:** Інтеграція модуля штучного інтелекту та організація командної роботи над застосунком.

Невеликий демонстраційний вебзастосунок для визначення тональності українських відгуків: позитивний або негативний. **Застосунок не використовує приватних API-ключів.** Модель Multinomial Naive Bayes тренується в Google Colab, експортується у JSON і виконується у браузері через JavaScript. Дані синтетичні й модель не призначена для реальних бізнес-рішень.

## Структура

- `notebooks/train_sentiment.ipynb` — навчання й експорт моделі в Colab
- `train_model.py` — той самий навчальний код для локального запуску
- `site/index.html` — сторінка
- `site/app.js` — завантаження й використання моделі в браузері
- `site/model.json` — навчальна модель (можна замінити версією, навченою в Colab)
- `tests/test_site.py` — перевірки
- `.github/workflows/pages.yml` — перевірка та автоматичний деплой на GitHub Pages

## 1. Навчити модель у Google Colab

1. Відкрийте [Google Colab](https://colab.research.google.com/).
2. Завантажте `notebooks/train_sentiment.ipynb` через **File → Upload notebook**.
3. Натисніть **Runtime → Run all**.
4. Завантажте отриманий `model.json` на комп'ютер.
5. Замініть ним `site/model.json` у проєкті й зробіть commit.

## 2. Запустити локально

Потрібний Python, але **не потрібен жоден Python-пакет** для роботи сайту:

```bash
python -m http.server 8000 -d site
```

Перейдіть у браузері на http://localhost:8000 . Не відкривайте `index.html` подвійним кліком: браузер може блокувати завантаження JSON із `file://`.

## 3. Розмістити на GitHub

Створіть публічний GitHub-репозиторій `sr2-ai-review-analyzer`, а тоді в папці проєкту виконайте:

```bash
git init
git add .
git commit -m "Initial AI review analyzer"
git branch -M main
git remote add origin https://github.com/USERNAME/sr2-ai-review-analyzer.git
git push -u origin main
```

Замініть `USERNAME` на свій GitHub username; Git попросить авторизуватися, наприклад через GitHub CLI. У **Settings → Pages → Build and deployment → Source** виберіть **GitHub Actions**. Перевірте **Actions** — `Test and deploy site` повинен завершитися успішно.

URL сайту буде приблизно `https://USERNAME.github.io/sr2-ai-review-analyzer/`.

## 4. Командна робота

У **Settings → Collaborators** додайте учасника. Нехай учасник створить гілку `feature/readme`, змінить README, зробить push і відкриє **Pull Request**. Після перевірки змерджіть PR у `main`. GitHub Actions повторно перевірить та опублікує сайт.

## Скриншоти для звіту

1. Навчання моделі в Google Colab — вивід тестової точності та створення `model.json`.
2. Вебсайт — введений відгук та два результати ймовірності.
3. Pull Request — видно автора, змінені файли й факт merge.
4. GitHub Actions — успішне тестування та деплой.
5. (За бажанням) Публічна адреса GitHub Pages.

## Обмеження

- Навчальних прикладів небагато, вони синтетичні, тому якість на довільних реальних відгуках не гарантується.
- Для прикладів на кшталт «не поганий» модель може помилятися.
- Відсотки — математичні оцінки маленької моделі, не калібрована впевненість.
