"use strict";

let model = null;
const button = document.getElementById("analyze");
const status = document.getElementById("status");
const result = document.getElementById("result");

// Match CountVectorizer's default Unicode word tokenizer (words of 2+ characters).
function tokenize(text) {
  return (text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []);
}

function infer(text, trainedModel) {
  const counts = new Map();
  for (const token of tokenize(text)) {
    const index = trainedModel.vocabulary[token];
    if (index !== undefined) counts.set(index, (counts.get(index) || 0) + 1);
  }
  if (!counts.size) return null;

  const scores = trainedModel.classes.map((_, classIndex) => {
    let score = trainedModel.class_log_prior[classIndex];
    for (const [index, amount] of counts) {
      score += amount * trainedModel.feature_log_prob[classIndex][index];
    }
    return score;
  });
  const maxScore = Math.max(...scores);
  const exponentials = scores.map(score => Math.exp(score - maxScore));
  const total = exponentials.reduce((sum, score) => sum + score, 0);
  return Object.fromEntries(trainedModel.classes.map((label, i) => [label, exponentials[i] / total]));
}

function showPercentage(id, value) {
  document.getElementById(`${id}-text`).textContent = `${(value * 100).toFixed(1)}%`;
  document.getElementById(`${id}-fill`).style.width = `${(value * 100).toFixed(1)}%`;
}

button.addEventListener("click", () => {
  const text = document.getElementById("review").value.trim();
  result.hidden = true;
  if (!text) {
    status.textContent = "Спочатку введіть відгук.";
    return;
  }
  const probabilities = infer(text, model);
  if (!probabilities) {
    status.textContent = "Не вистачає знайомих моделі слів. Спробуйте інший відгук.";
    return;
  }
  showPercentage("positive", probabilities.positive || 0);
  showPercentage("negative", probabilities.negative || 0);
  document.getElementById("decision").textContent =
    (probabilities.positive >= probabilities.negative) ? "Результат: позитивний відгук" : "Результат: негативний відгук";
  status.textContent = "Аналіз завершено.";
  result.hidden = false;
});

fetch("model.json")
  .then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(loaded => {
    model = loaded;
    button.disabled = false;
    status.textContent = `Модель готова (${model.training_examples} навчальних прикладів).`;
  })
  .catch(() => {
    status.textContent = "Не вдалося завантажити модель. Запустіть сайт через HTTP-сервер або GitHub Pages.";
  });
