# 🎌 MangaTR-Studio — Yapay Zeka Destekli Türkçe Manga Çeviri, Balon Temizleme & Dizgi Stüdyosu

**MangaTR-Studio**, İngilizce manga sayfalarını uçtan uca (veya kullanıcı kontrollü masaüstü arayüzüyle) **tespit eden, satır bazlı OCR ile okuyan, bağlama uygun doğal Türkçeye çeviren, konuşma balonlarını çizgilere zarar vermeden temizleyen (inpaint) ve Türkçe karakter destekli (`Ç, Ğ, İ, Ö, Ş, Ü`) profesyonel manga dizgisi (typesetting) yapan** açık kaynaklı bir masaüstü uygulamasıdır.

> 💡 **API Anahtarı Gerekir mi?**
> **Hayır!** MangaTR-Studio kutudan çıktığı haliyle **hiçbir ücretli API anahtarına ihtiyaç duymadan (`free_web` / `auto` moduyla) %100 ücretsiz** çalışır.
> İsteğe bağlı olarak kendi **Google Gemini API**, **OpenAI / DeepSeek / OpenRouter API** anahtarınızı veya tamamen çevrimdışı çalışan **Yerel LLM (Ollama)** modelinizi bağlayabilirsiniz.

---

## ✨ Temel Özellikler

1. **🎯 Akıllı Konuşma Balonu & Satır Tespiti (`app/engines/detector.py`)**
   - `Comic Text Detector` tabanlı derin öğrenme modeli ile konuşma balonlarını, anlatıcı kutularını ve balon dışı (floating) metinleri tespit eder.
   - **Duplicate (Çift) Blok Temizliği**: İç içe geçmiş veya üst üste binen mükerrer kutuları IoU ve kapsama (containment) analiziyle tek bir temiz blokta birleştirir.
   - **Maskeden Eksik Blok Kurtarma**: Dedektörün kutu olarak kaçırdığı ama metin maskesinde yakaladığı küçük balonları (ör. *"I SEE..."*) otomatik kurtarır.
   - **Manga Okuma Sırası**: Sağdan sola (`manga_rtl`) veya soldan sağa (`comic_ltr`) doğru okuma sırasına göre numaralandırır.

2. **🔍 Satır Bazlı (Line-Level) OCR & Manga Metin Normalizasyonu (`app/engines/ocr_engine.py`)**
   - Tüm balonu tek seferde okumak yerine `lines` poligonlarını satır satır kırpar, balon çerçevesi kalıntılarını temizler ve 2.2x büyüterek **EasyOCR** ile okur.
   - Satır sonu tirelemelerini (`CAP-\nTAIN` -> `CAPTAIN`) ve yaygın OCR karakter hatalarını (`1` / `I`, `0` / `O`) otomatik düzeltir.

3. **🌍 Çoklu Çeviri Motoru & Bölüm Hafızalı Sözlük (`app/engines/translator.py`)**
   - **Ücretsiz Manga Yerelleştirici (`free_web` - Varsayılan)**: API anahtarı gerektirmez; manga konuşma kalıplarını ve deyimleri doğal Türkçeye uyarlar.
   - **Yerel LLM (`local_llm` - Ollama)**: `qwen2.5:7b` vb. yerel modellerle %100 çevrimdışı, sayfa bağlamını koruyan çeviri yapar.
   - **Google Gemini API (`gemini_api` - Opsiyonel)**: `GEMINI_API_KEY` girildiğinde tüm sayfa diyaloglarını tek seferde bağlama duyarlı çevirir.
   - **OpenAI / DeepSeek / OpenRouter API (`openai_api` - Opsiyonel)**: OpenAI uyumlu herhangi bir LLM API'si ile çalışır.
   - **Proje Terimler Sözlüğü (`glossary.json`)**: 1'den 400+ bölüme kadar özel isimlerin (*Whale Island -> Balina Adası*, *Hunter -> Avcı*) tüm bölümlerde tutarlı çevrilmesini sağlar.

4. **🧹 Akıllı Balon Temizleme & Hibrit Inpainting (`app/engines/cleaner.py`)**
   - **Beyaz/Düz Konuşma Balonları**: Balonun dış siyah kontur çizgilerini koruyarak iç metin bölgesini pürüzsüz balon arka plan rengiyle temizler.
   - **Karmaşık / Desenli Arka Planlar**: Balon dışı metinleri OpenCV Telea/NS inpainting ve yumuşak geçiş (feathering) ile dokuyu bozmadan siler.

5. **✍️ Türkçe Manga Dizgi Motoru (`app/engines/typesetter.py`)**
   - Tam Türkçe büyük/küçük harf dönüşümü (`i -> İ`, `ı -> I`, `ğ -> Ğ`, `ş -> Ş`, `ç -> Ç`, `ö -> Ö`, `ü -> Ü`).
   - Balonun güvenli iç alanına göre dengeli satır kırma (balanced line wrapping) ve dinamik font boyutu hesaplama.
   - Bağırma/ünlem (`!!`, `?!`) balonlarını otomatik algılayarak uygun font stili seçme ve desenli arka planlarda otomatik beyaz kontur (stroke halo) ekleme.

6. **🖥️ Masaüstü Stüdyo Arayüzü & Toplu Bölüm (Batch) İşleme (`app/ui/main_window.py`)**
   - Görsel üzerinde balon seçme, kutu boyutunu fareyle sürükleyerek düzenleme, metni değiştirip **anında (0.1 sn)** yeniden basma.
   - **Yan Yana (Orijinal | Türkçe)**, **Temizlenmiş (Inpaint)** ve **Tespit Kutuları** görünüm modları.
   - Tek tıklamayla çoklu bölüm klasörlerini (`Chapter_001` ... `Chapter_400`) kaldığı yerden devam etme (resume) desteğiyle toplu çevirme.

---

## 🚀 Kurulum

### Gereksinimler
- **İşletim Sistemi**: Windows 10 / 11 (Linux ve macOS üzerinde de Python ile çalışır)
- **Python**: Python 3.10 veya 3.11 (64-bit)
- **GPU (Önerilen)**: NVIDIA CUDA destekli ekran kartı (Yoksa otomatik olarak CPU modunda çalışır)

### Yöntem 1: Windows'ta Tek Tıkla Otomatik Kurulum (Önerilen)

Depoyu klonladıktan sonra **`setup.bat`** dosyasına çift tıklayın:

```bat
git clone https://github.com/yogruce/MangaTR-Studio.git
cd MangaTR-Studio
setup.bat
```

`setup.bat` otomatik olarak:
1. `.venv` sanal ortamını oluşturur,
2. `requirements.txt` içindeki paketleri yükler,
3. `setup_project.py` betiğini çalıştırarak `comictextdetector.pt` (~80 MB) modelini indirir ve Türkçe fontları hazırlar.

### Yöntem 2: Manuel Kurulum (Terminal / PowerShell)

```powershell
# 1. Sanal ortam oluşturun ve etkinleştirin
python -m venv .venv
.\.venv\Scripts\activate

# 2. (Önerilen - NVIDIA GPU varsa) CUDA destekli PyTorch kurun:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Model ağırlıklarını (comictextdetector.pt) ve klasörleri hazırlayın
python setup_project.py
```

> **Not:** `setup_project.py` çalıştırılmasa bile, uygulamayı ilk kez başlattığınızda `comictextdetector.pt` modeli eksikse otomatik olarak indirilir.

---

## 🔑 API Ayarları (Opsiyonel)

MangaTR-Studio **varsayılan olarak herhangi bir API anahtarı istemez**. Ancak kendi API anahtarınızı kullanmak isterseniz iki farklı yöntemle tanımlayabilirsiniz:

### Seçenek A: Masaüstü Arayüzünden ("🔑 API Ayarları" Butonu)
1. Uygulamayı başlatın (`MangaTR-Studio.bat` veya `python app/main.py`).
2. Üst araç çubuğundaki **"🔑 API Ayarları"** butonuna tıklayın.
3. Kullanmak istediğiniz API anahtarını (`GEMINI_API_KEY` veya `OPENAI_API_KEY`) girip **"💾 Kaydet (.env)"** butonuna basın.
4. Sağ üstteki **Çeviri Motoru** kutusundan `auto`, `gemini_api`, `openai_api` veya `local_llm` seçin.

### Seçenek B: `.env` Dosyası ile
Proje ana dizinindeki `.env.example` dosyasını `.env` adıyla kopyalayın:

```powershell
copy .env.example .env
```

Ardından `.env` dosyasını bir metin editörüyle açıp kullanmak istediğiniz servisin anahtarını girin:

```env
# Seçenekler: auto | free_web | local_llm | gemini_api | openai_api
MANGATR_BACKEND=auto

# 1. Google Gemini API (Opsiyonel - https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=AIzaSyYourApiKeyHere
GEMINI_MODEL=gemini-2.5-flash

# 2. OpenAI / DeepSeek / OpenRouter API (Opsiyonel)
OPENAI_API_KEY=sk-YourApiKeyHere
OPENAI_COMPAT_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4o-mini

# 3. Yerel Ollama LLM (Opsiyonel & Ücretsiz Çevrimdışı)
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b
```

> 🔒 **Güvenlik Notu:** `.env` dosyası `.gitignore` içinde tanımlıdır; kişisel API anahtarlarınız hiçbir zaman Git/GitHub'a yüklenmez.

---

## 🖥️ Kullanım

### 1. Masaüstü Arayüzünü (GUI) Başlatma
`MangaTR-Studio.bat` dosyasına çift tıklayın veya terminalden:
```powershell
python app/main.py
```

### 2. Komut Satırı (CLI) ile Tek Sayfa Çevirme
```powershell
python app/main.py --cli -i input/test.jpg -o output/
```
Çıktılar `output/` klasörüne kaydedilir:
- `output/test_tr.jpg` — Türkçe dizgisi tamamlanmış final manga sayfası
- `output/test_cleaned.jpg` — Konuşma balonları temizlenmiş (inpaint) sayfa
- `output/test_detected.jpg` — Okuma sırasına göre numaralandırılmış tespit kutuları
- `output/test_tr.json` — Tüm balonların koordinatları, satırları, İngilizce OCR metinleri ve Türkçe çevirileri

### 3. Komut Satırı (CLI) ile Çoklu Bölüm (Batch) Çevirme
```powershell
python app/main.py --cli --batch -i "C:/Manga/HunterXHunter" -o "C:/Manga/HunterXHunter_TR"
```

### 4. Sadece Tespit + Satır Bazlı OCR + Çeviri Testi (`text_detection.py`)
```powershell
python text_detection.py -i input/test.jpg -o output/
```

---

## 📁 Proje Klasör Yapısı

```text
MangaTR-Studio/
├── app/
│   ├── config.py                 # Merkezi yapılandırma ve .env yükleyici
│   ├── main.py                   # GUI ve CLI ana giriş noktası
│   ├── core/
│   │   ├── project.py            # Bölüm hafızası ve Terimler Sözlüğü (glossary.json) yöneticisi
│   │   ├── pipeline.py           # Uçtan uca sayfa işleme hattı + anlık yeniden dizgi
│   │   └── batch_processor.py    # 1-400+ bölüm toplu çeviri ve devam ettirme (resume) motoru
│   ├── engines/
│   │   ├── detector.py           # Comic Text Detector + Duplicate temizleme + Maskeden kurtarma
│   │   ├── ocr_engine.py         # Satır bazlı (Line-Level) EasyOCR + Manga metin düzeltici
│   │   ├── translator.py         # Çoklu çeviri motoru (Free Web / Ollama / Gemini / OpenAI)
│   │   ├── cleaner.py            # Akıllı balon temizleme + Hibrit inpainting
│   │   ├── inpainter.py          # OpenCV Telea / NS / LaMa ONNX sarmalayıcı
│   │   └── typesetter.py         # Türkçe karakter destekli otomatik dizgi ve satır dengeleme
│   ├── ui/
│   │   └── main_window.py        # Tkinter Koyu Tema Masaüstü Stüdyo Arayüzü
│   └── utils/
│       ├── json_utils.py         # Numpy -> Python güvenli JSON serileştirici
│       └── text_normalizer.py    # OCR temizleme ve Türkçe büyük/küçük harf araçları
├── fonts/                        # Açık kaynaklı (OFL) Türkçe destekli manga fontları
├── input/                        # Örnek test manga sayfası (test.jpg)
├── output/                       # Çevrilen sayfaların ve JSON çıktılarının kaydedildiği klasör
├── projects/default/             # Varsayılan proje sözlüğü (glossary.json)
├── tests/                        # Birim ve entegrasyon testleri (test_pipeline.py)
├── vendor/comic-text-detector/   # Metin ve balon tespit modeli altyapısı
├── .env.example                  # Opsiyonel API anahtarları için örnek şablon
├── .gitignore                    # Git hariç tutma kuralları (.venv, .env, *.pt vb.)
├── requirements.txt              # Python paket bağımlılıkları
├── setup_project.py              # Model (comictextdetector.pt) ve font otomatik kurulum betiği
├── setup.bat                     # Windows tek tıkla kurulum dosyası
├── MangaTR-Studio.bat            # Windows tek tıkla başlatıcı
└── build_exe.py                  # PyInstaller ile taşınabilir .exe oluşturma betiği
```

---

## 🧪 Testleri Çalıştırma

```powershell
python tests/test_pipeline.py
```

## 📦 Taşınabilir `.exe` Oluşturma

Projeyi Python kurulu olmayan bilgisayarlarda çalıştırmak üzere paketlemek için:

```powershell
pip install pyinstaller
python build_exe.py
```
Paketlenen uygulama `dist/MangaTR-Studio/` klasöründe oluşturulur.
