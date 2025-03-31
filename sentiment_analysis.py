import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')

try:
    tr_stopwords = stopwords.words('turkish')
except:
    print("Türkçe stopwords bulunamadı, boş liste kullanılıyor.")  
    tr_stopwords = []

custom_stopwords = ['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri',
                    'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye',
                    'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez',
                    'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye',
                    'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
tr_stopwords = list(set(tr_stopwords + custom_stopwords))

def temizle_metin(metin):
    metin = metin.lower()
    metin = re.sub(r'<.*?>', '', metin)
    metin = re.sub(r'[^a-zçğıöşü0-9\s]', '', metin)
    metin = re.sub(r'\d+', '', metin)
    metin = ' '.join([kelime for kelime in metin.split() if kelime not in tr_stopwords])
    return metin.strip()

df = pd.read_csv("dataset/magaza_yorumlari_duygu_analizi.csv", encoding='utf-16')
df.columns = ["yorum", "durum"]

df = df.dropna()
df["temiz_yorum"] = df["yorum"].apply(temizle_metin)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["temiz_yorum"])
y = df["durum"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

def tahmin_yap(metin):
    temiz_metin = temizle_metin(metin)
    metin_vector = vectorizer.transform([temiz_metin])
    tahmin = model.predict(metin_vector)[0]
    return tahmin

sns.countplot(x=df['durum'])
plt.title("Sınıf Dağılımı")
plt.show()

print("\nKendi yorumunuzu girin ('q' ile çıkış yapabilirsiniz):")
while True:
    user_input = input("Yorum: ")
    if user_input.lower() == 'q':
        break
    print("Tahmin Edilen Duygu:", tahmin_yap(user_input))
