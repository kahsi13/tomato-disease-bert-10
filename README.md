# 🍅 Tomato Disease BERT - 10 Sınıf

Bu proje, domates yaprak hastalıklarını Türkçe metin açıklamalar üzerinden sınıflandırmak için özel olarak eğitilmiş bir **BERT tabanlı NLP modeli** içerir.

## 🧠 Model Özellikleri

- **Model Tabanı:** `dbmdz/bert-base-turkish-cased`
- **Sınıf Sayısı:** 10 (ör. Erken Yaprak Yanıklığı, Mozaik Virüsü, Kırmızı Örümcek vs.)
- **Doğruluk:** ~78% F1 skoru
- **Veri Seti:** Dengelenmiş, her sınıf için 200 örnek içeren özel eğitim verisi
- **Eğitim:** 100 epoch + early stopping (patience=5), learning rate: 2e-5

## 🚀 API Özeti

Model, FastAPI ile paketlenmiştir ve `/predict` endpoint’i üzerinden POST isteği ile tahmin döner:

```json
POST /predict
{
  "text": "Yapraklarda kahverengi lekeler oluşmuş, kenarları kurumuş."
}
```

Yanıt:
```json
{
  "label": "Erken Yaprak Yanıklığı (Alternaria solani)",
  "confidence": 0.9874
}
```

## 🔗 Bağlantılar

- 🤗 [Hugging Face Model Sayfası](https://huggingface.co/Kahsi13/tomato-bert-10)
- 💻 [GitHub Repo](https://github.com/kahsi13/tomato-disease-bert-10)

## 📂 Proje Yapısı

```
tomato-disease-bert-10/
├── data/
│   └── bert_dataset_top10_balanced.csv
├── model/
│   ├── model.safetensors
│   ├── config.json
│   └── ...
├── scripts/
│   ├── main.py
│   ├── test.py
│   └── requirements.txt
```

## 👨‍🔬 Geliştirici Notu

Bu model, tarım uygulamaları ve mobil tanı sistemleri için ideal bir metin sınıflandırma örneğidir.  
Gerekli API entegrasyonları ve Flutter bağlantısı hazırdır.

---

Made with ❤️ by [Kahsi13](https://github.com/kahsi13)
