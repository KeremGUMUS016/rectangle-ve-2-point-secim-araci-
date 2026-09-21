# 🌱 Fidan Pose Labeler

**Fidan Pose Labeler**, robotik fidan hasadı projeleri için geliştirilmiş, masaüstünde çalışan bir **YOLO Pose etiketleme ve aktif öğrenme uygulamasıdır**.

Uygulama, her güvenle kavranabilir fidan için:

- Bir adet bounding box
- `sap_ucu` keypoint’i
- `aci_noktasi` keypoint’i

oluşturur. Manuel etiketleme, YOLO Pose eğitimi, model ile ön etiketleme ve insan onayı aynı uygulama içerisinde yapılabilir.

> Bu uygulama özellikle bant üzerindeki fidanların robot koluyla kavranacağı senaryolar için geliştirilmiştir.

---

## Özellikler

- PySide6 tabanlı masaüstü arayüzü
- YOLO Pose formatında etiketleme
- Her fidan için bounding box ve iki keypoint
- Otomatik etiketleme sırası:
  1. Bounding box
  2. `sap_ucu`
  3. `aci_noktasi`
- Sağ tuşla bounding box köşelerini ve keypoint’leri düzenleme
- Undo/redo desteği
- Touchpad ile kaydırma ve yakınlaştırma
- Atomik JSON otomatik kayıt
- YOLO Pose veri seti dışa aktarma
- Yerel NVIDIA GPU üzerinde YOLO11m Pose eğitimi
- Eğitilmiş modelle toplu ön etiketleme
- İnsan tarafından kontrol edilen etiketlerin eğitime alınması
- Boş negatif görüntü desteği
- Eğitim verisine yalnızca incelenmiş görüntülerin eklenmesi

---

## Etiket Renkleri

| Renk | Anlamı |
|---|---|
| Mor | Model tarafından oluşturulmuş, henüz kontrol edilmemiş tahmin |
| Sarı | Kullanıcı tarafından düzenlenmiş model tahmini |
| Manuel etiket rengi | Kullanıcı tarafından doğrudan çizilmiş etiket |
| Onaylanmış | `Onayla ve Sonraki` işleminden sonra eğitime uygun etiket |

Bir model tahminini düzenlemek tek başına görüntüyü onaylamaz. Kontrol tamamlandıktan sonra **`Onayla ve Sonraki`** düğmesi veya **`Ctrl+Enter`** kullanılmalıdır.

---

## Etiket Yapısı

Uygulamada yalnızca bir sınıf bulunmaktadır:

```text
class_id = 0
class_name = fidan
```

Keypoint sırası sabittir:

```text
0 = sap_ucu
1 = aci_noktasi
```

Her nesnenin YOLO Pose satırı şu yapıdadır:

```text
class x_center y_center width height sap_x sap_y sap_visibility aci_x aci_y aci_visibility
```

Örnek:

```text
0 0.51200000 0.43800000 0.22000000 0.31000000 0.49500000 0.57500000 2 0.53000000 0.43000000 2
```

---

## Etiketleme Kuralları

Modelin yalnızca robot tarafından güvenle kavranabilir fidanları öğrenmesi hedeflenmektedir.

Bir fidanı etiketlemek için:

- Sap ucu net biçimde görünmelidir.
- Fidanın yönü anlaşılabilmelidir.
- Hangi yaprakların aynı fidana ait olduğu belirlenebilmelidir.
- Fidan üstten fiziksel olarak erişilebilir olmalıdır.
- Bounding box yalnızca aynı fidana ait görünür kısımları kapsamalıdır.

Şunları etiketlemeyin:

- Sap ucu görünmeyen fidanlar
- Diğer fidanların altında kalanlar
- Birbirinden ayrılamayan yoğun kümeler
- Hangi yaprakların aynı fidana ait olduğu belirsiz örnekler
- Görünmeyen kısmı tahmin edilmek zorunda olan fidanlar
- Robot tarafından güvenle kavranamayacak örnekler

Bir görüntüde çok sayıda fidan bulunmasına rağmen yalnızca birkaç tanesi güvenle kavranabiliyorsa sadece güvenli olanlar etiketlenmelidir.

---

## Sistem Gereksinimleri

### Önerilen sistem

- Windows 10 veya Windows 11
- NVIDIA ekran kartı
- CUDA destekli PyTorch
- Python 3.11, 3.12 veya uyumlu daha yeni bir sürüm
- En az 8 GB RAM
- YOLO11m Pose eğitimi için tercihen en az 8 GB VRAM

Uygulama CPU ile arayüz ve manuel etiketleme amacıyla kullanılabilir. Ancak yerel eğitim ve hızlı ön etiketleme için CUDA destekli NVIDIA GPU önerilir.

---

## Kurulum

### 1. Projeyi indirin

GitHub üzerindeki **Code → Download ZIP** seçeneğini kullanabilirsiniz.

### 2. Sanal ortam oluşturun

```powershell
py -m venv .venv
```

### 3. Sanal ortamı etkinleştirin

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell çalıştırma ilkesi nedeniyle hata alırsanız:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. PyTorch kurun

PyTorch’u ekran kartınıza ve CUDA sürümünüze uygun biçimde kurun.

Kurulumdan sonra CUDA kontrolü:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok')"
```

Beklenen örnek çıktı:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 4070
```

### 5. Diğer bağımlılıkları kurun

```powershell
pip install ultralytics PySide6
```

---

## Uygulamayı Çalıştırma

```powershell
.\.venv\Scripts\python.exe .\fidan_pose_labeler.py
```

Windows üzerinde uygulamayı çift tıklayarak açmak için şu içerikle bir `fidan_pose_labeler_ac.bat` dosyası oluşturabilirsiniz:

```bat
@echo off
setlocal

set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "APP=%~dp0fidan_pose_labeler.py"

if not exist "%PYTHON%" (
    echo HATA: Sanal ortam Python'u bulunamadi:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo HATA: Uygulama dosyasi bulunamadi:
    echo %APP%
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON%" "%APP%"

if errorlevel 1 (
    echo.
    echo Uygulama bir hatayla kapandi.
    pause
)

endlocal
```

BAT dosyası, `.venv` klasörü ve `fidan_pose_labeler.py` aynı proje klasöründe bulunmalıdır.

---

## Kullanım

### 1. Görüntü klasörünü açın

Uygulamayı açtıktan sonra **Klasör Aç** düğmesiyle görüntülerin bulunduğu klasörü seçin.

Desteklenen formatlar:

- JPG / JPEG
- PNG
- BMP
- WEBP
- TIFF

### 2. Manuel etiket oluşturun

Her fidan için:

1. Sol tuşla bounding box çizin.
2. `sap_ucu` noktasına tıklayın.
3. `aci_noktasi` noktasına tıklayın.

Bounding box, aynı fidana ait tüm görünür kısımları sıkıca çevrelemelidir.

### 3. Etiketi düzenleyin

Sağ fare tuşuyla sürükleyerek:

- Bounding box köşelerini
- `sap_ucu` noktasını
- `aci_noktasi` noktasını

düzenleyebilirsiniz.

### 4. Görüntüyü onaylayın

Görüntüdeki tüm etiketleri kontrol ettikten sonra:

```text
Ctrl + Enter
```

kısayolunu veya **Onayla ve Sonraki** düğmesini kullanın.

Bu işlem:

- Model tahminlerini insan onaylı hâle getirir.
- Görüntüyü incelenmiş olarak işaretler.
- Etiketleri sonraki eğitime uygun hâle getirir.
- Sonraki görüntüye geçer.

Normal sağ ok veya `D` ile geçilen onaylanmamış model tahminleri eğitime alınmaz.

---

## Boş Negatif Görüntüler

Bir görüntüde güvenle kavranabilir fidan yoksa:

1. Yanlış model tahminlerinin tamamını silin.
2. Görüntüde hiç nesne bırakmayın.
3. **Onayla ve Sonraki** düğmesine basın.

Bu görüntü bilinçli bir negatif örnek olarak eğitime dahil edilir.

Dokunulmamış boş görüntüler ise otomatik olarak eğitim verisine eklenmez.

---

## Klavye Kısayolları

| Kısayol | İşlem |
|---|---|
| `B` | Yeni bounding box çizme modu |
| `S` | Seçim modu |
| `K` | Seçili nesnenin keypoint’lerini yeniden yerleştirme |
| `A` veya `Sol Ok` | Önceki görüntü |
| `D` veya `Sağ Ok` | Sonraki görüntü |
| `Ctrl+Enter` | Görüntüyü onayla ve sonraki görüntüye geç |
| `Delete` / `Backspace` | Seçili nesneyi sil |
| `X` | Son oluşturulan nesneyi sil |
| `Ctrl+Z` | Geri al |
| `Ctrl+Y` | İleri al |
| `Escape` | Aktif çizim işlemini iptal et |
| `Ctrl++` | Yakınlaştır |
| `Ctrl+-` | Uzaklaştır |
| `Ctrl+0` | Yakınlaştırmayı sıfırla |
| `Ctrl+S` | Mevcut görüntüyü kaydet |
| `Ctrl+E` | YOLO Pose veri setini dışa aktar |

---

## Otomatik Kayıt

Etiketler görüntü klasörünün içerisinde şu dosyaya otomatik olarak kaydedilir:

```text
.fidan_pose_annotations.json
```

Bu dosya şunları saklar:

- Bounding box koordinatları
- Keypoint koordinatları
- Etiketin manuel veya model kaynaklı olması
- İnsan tarafından kontrol edilip edilmediği
- Görüntünün incelenmiş olup olmadığı
- Kullanılan model yolu
- Model güven değerleri

> Bu dosyayı silmeyin. Etiketleme çalışmasının ana kayıt dosyasıdır.

Kayıt işlemi geçici dosya üzerinden atomik olarak yapılır. Böylece uygulamanın beklenmedik biçimde kapanması durumunda veri kaybı riski azaltılır.

---

## YOLO11 Pose Eğitimi

Uygulama içindeki **YOLO11m Eğit** düğmesi, insan tarafından onaylanmış verilerle yeni bir eğitim başlatır.

Varsayılan eğitim ayarları:

```text
Model: yolo11m-pose.pt
Epoch: 300
Image size: 1024
Batch: 2
Device: CUDA 0
Workers: 0
AMP: False
Cache: False
Patience: 80
Mosaic: 0
```

Ek veri artırma ayarları:

```text
degrees=0.0
translate=0.05
scale=0.20
shear=0.0
perspective=0.0
fliplr=0.5
flipud=0.0
```

Eğitim her seferinde temiz `yolo11m-pose.pt` ağırlıklarından başlar.

Eğitim sonuçları görüntü klasörünün altında oluşturulur:

```text
fidan_pose_training/
└── runs/
    └── round_TARIH_SAAT/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        ├── results.png
        ├── train_batch0.jpg
        └── val_batch0_pred.jpg
```

RTX ekran kartında VRAM hatası alınırsa kod içerisindeki:

```python
TRAIN_BATCH = 2
```

değeri şu şekilde değiştirilebilir:

```python
TRAIN_BATCH = 1
```

---

## Ön Etiketleme

Bir eğitim tamamlandıktan sonra `best.pt` dosyası uygulamaya yüklenebilir.

Kullanılabilecek işlemler:

- **Bu Fotoğrafı Ön Etiketle**
- **Etiketsizleri Ön Etiketle**

Ön etiketleme ayarları:

```text
imgsz=1024
iou=0.50
device=0
conf=arayüzdeki güven değeri
```

Araç çubuğundaki güven değeri yalnızca ön etiketlemeyi etkiler; eğitimi etkilemez.

Örnek:

| Güven | Sonuç |
|---:|---|
| `0.35` | Daha fazla tahmin, daha fazla yanlış pozitif ihtimali |
| `0.50` | Dengeli ve daha seçici başlangıç |
| `0.60` | Daha az tahmin, daha yüksek precision eğilimi |

Önceden oluşturulmuş tahminler güven değeri değiştirildiğinde otomatik olarak filtrelenmez. Görüntünün yeni eşikle tekrar ön etiketlenmesi gerekir.

---

## Aktif Öğrenme Akışı

Önerilen çalışma düzeni:

1. İlk görüntü grubunu manuel olarak etiketleyin.
2. YOLO11m Pose modelini eğitin.
3. Yeni modeli uygulamaya yükleyin.
4. Etiketsiz görüntüleri modelle ön etiketleyin.
5. Mor tahminleri kontrol edin.
6. Yanlış tahminleri silin.
7. Hatalı kutuları ve keypoint’leri düzeltin.
8. Eksik fidanları manuel ekleyin.
9. `Ctrl+Enter` ile görüntüyü onaylayın.
10. Düzeltilmiş veriyle modeli yeniden eğitin.

Bu yöntem, bütün görüntüleri sıfırdan manuel etiketlemekten daha hızlıdır.

---

## Eğitim Verisine Neler Dahil Edilir?

Eğitime dahil edilir:

- Tamamlanmış manuel etiketler
- İnsan tarafından onaylanmış model tahminleri
- İnsan tarafından düzeltilip onaylanmış model tahminleri
- Bilinçli olarak onaylanmış boş negatif görüntüler

Eğitime dahil edilmez:

- Dokunulmamış görüntüler
- Eksik keypoint içeren nesneler
- Henüz onaylanmamış model tahminleri
- Yalnızca model tarafından üretilmiş ancak insan kontrolünden geçmemiş sonuçlar

---

## RGB-D Kullanımı

Mevcut YOLO Pose modeli RGB görüntü üzerinde çalışır. Depth görüntüsü doğrudan YOLO11m Pose girişine verilmez.

Önerilen robotik entegrasyon:

1. YOLO modeli RGB görüntüde `sap_ucu` pikselini bulur.
2. Aynı pikselin hizalanmış depth değeri okunur.
3. Kamera intrinsic değerleriyle piksel 3B kamera koordinatına dönüştürülür.
4. Kamera–robot kalibrasyonuyla robot koordinatına aktarılır.
5. `sap_ucu → aci_noktasi` doğrultusu kavrama açısının hesaplanmasında kullanılır.
6. Depth verisi güvenli yaklaşma ve üst üste binme kontrolü için ek filtre olarak değerlendirilir.

---

## Dosya Yapısı

Önerilen proje yapısı:

```text
fidan-pose-labeler/
├── fidan_pose_labeler.py
├── fidan_pose_labeler_ac.bat
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

Görüntü veri setleri, sanal ortam ve model ağırlıkları GitHub deposuna eklenmemelidir.

Önerilen `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# Application annotations
.fidan_pose_annotations.json
.fidan_pose_annotations.json.tmp

# Training artifacts
fidan_pose_training/
runs/

# Model weights
*.pt
*.onnx
*.engine

# Datasets and media
*.jpg
*.jpeg
*.png
*.bmp
*.webp
*.tif
*.tiff
*.mp4
*.avi
*.mov

# IDE
.vscode/
.idea/

# Operating system
.DS_Store
Thumbs.db
```

Örnek `requirements.txt`:

```text
PySide6
ultralytics
```

> PyTorch kurulumu CUDA sürümüne bağlı olduğu için `requirements.txt` içerisinde sabitlenmemiştir. Kullanıcılar önce kendi sistemlerine uygun CUDA destekli PyTorch sürümünü kurmalıdır.

---

## Gizlilik ve Veri

Uygulama yerel olarak çalışır.

- Görüntüler otomatik olarak bir bulut hizmetine yüklenmez.
- Etiketler yerel JSON dosyasında tutulur.
- Eğitim yerel GPU üzerinde yapılır.
- Ön etiketleme yerel olarak çalışır.
- Roboflow veya başka bir bulut servisi kullanımı zorunlu değildir.

---

## Bilinen Sınırlamalar

- Uygulama mevcut hâliyle tek sınıflıdır: `fidan`.
- Keypoint sayısı sabittir ve iki tanedir.
- Eğitim ayrımı rastgele yapılır.
- Final değerlendirmede art arda çekilmiş benzer karelerin train/validation/test arasında karışmaması için sahne veya çekim oturumu bazlı ayrım önerilir.
- RGB görüntüden anlaşılamayan fiziksel erişilebilirlik durumlarında depth veya ek robotik güvenlik kontrolü gerekir.
- `best.pt` oluşması modelin otomatik olarak güvenilir olduğu anlamına gelmez; ayrı test görüntüleri üzerinde doğrulama yapılmalıdır.

---

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır.

1. Depoyu fork edin.
2. Yeni bir branch oluşturun:

```bash
git checkout -b feature/yeni-ozellik
```

3. Değişikliklerinizi commit edin:

```bash
git commit -m "Yeni özellik eklendi"
```

4. Branch’i gönderin:

```bash
git push origin feature/yeni-ozellik
```

5. Pull Request oluşturun.

Özellikle şu katkılar faydalıdır:

- Çoklu sınıf desteği
- Keypoint şeması düzenleyicisi
- Sahne bazlı train/validation/test ayrımı
- Etiket kalite kontrol araçları
- COCO Keypoints içe/dışa aktarma
- Linux ve macOS testleri
- RGB-D görselleştirme
- Model metriklerinin uygulama içinde gösterilmesi

---

## Lisans

Projeyi açık kaynak paylaşacaksanız depoya bir lisans eklemeniz önerilir.

Geniş kullanım ve katkı için yaygın seçenek:

```text
MIT License
```

Ancak Ultralytics ve diğer bağımlılıkların kendi lisans koşulları ayrıca geçerlidir. Özellikle ticari kullanım veya kapalı kaynak dağıtım planlanıyorsa kullanılan tüm bağımlılıkların güncel lisans koşulları kontrol edilmelidir.

---

## Uyarı

Bu yazılım araştırma ve geliştirme amacıyla hazırlanmıştır. Gerçek robotik sistemde kullanılmadan önce:

- Ayrı test verisiyle doğrulanmalı
- Hatalı pozitif ve negatif durumları ölçülmeli
- Robot erişim sınırları uygulanmalı
- Depth doğrulaması yapılmalı
- Acil durdurma ve fiziksel güvenlik önlemleri eklenmeli
- İnsanların bulunduğu çalışma alanlarında bağımsız güvenlik sistemleri kullanılmalıdır

Model çıktıları tek başına fiziksel robot güvenlik mekanizması olarak değerlendirilmemelidir.

---

## Ekran Görüntüsü

Bu bölüme uygulamanızın ekran görüntüsünü ekleyebilirsiniz:

```markdown
![Fidan Pose Labeler](docs/screenshot.png)
```

Önerilen yapı:

```text
docs/
└── screenshot.png
```

---

## Teşekkürler

Bu proje; robotik hasat, aktif öğrenme, keypoint detection ve insan onaylı ön etiketleme çalışmalarını daha hızlı hâle getirmek amacıyla geliştirilmiştir.

Projeyi faydalı bulduysanız GitHub üzerinden yıldız verebilir, hata bildirebilir veya geliştirmelere katkıda bulunabilirsiniz. 🌱🤖
