# Recorder dataset pentru semne

Programul folosește camera doar pentru detectarea în timp real și salvează pentru fiecare cadru punctele 3D ale mâinilor detectate. Nu salvează imagini sau clipuri video. Pot fi detectate una sau două mâini. Datele rămân local pe calculator.

## Pornire

Instalează Python 3.10 sau 3.13, apoi în terminal:

```bash
python -m pip install -r requirements.txt
python hand_dataset_recorder.py --label salut
```

La prima pornire descarcă automat modelul oficial MediaPipe (`hand_landmarker.task`) lângă program. Dacă ai altă cameră, folosește `--camera 1`; pentru alt dosar de ieșire, `--out datele_mele`.

În cod se importă `cv2`, dar pachetul nu se numește `cv2`. Instalarea MediaPipe din `requirements.txt` instalează și OpenCV, care furnizează modulul `cv2`.

## Înregistrare

- `SPACE`: începe sau oprește o mostră pentru eticheta dată.
- `Q` sau `ESC`: închide programul.

Înregistrează mai multe mostre pentru fiecare semn, cu persoane, unghiuri, distanțe și condiții de lumină diferite. Pentru un semn cu mișcare, începe înainte de gest și oprește după ce s-a terminat.

## Fișiere rezultate

Pentru fiecare mostră se creează doar un fișier `.jsonl`, numerotat după etichetă (`salut1.jsonl`, `salut2.jsonl` etc.). Acesta conține eticheta, timpul/cadrul, mâna detectată și 21 de landmarks normalizați (`x/y/z`) plus coordonatele 3D `world_landmarks`. Când nu se detectează nicio mână, cadrul este păstrat cu `hands: []`.

Etichetele trebuie să fie aceleași de fiecare dată (de exemplu `salut`, `multumesc`, `da`). JSONL este potrivit pentru preprocesare ulterioară într-un model temporal.

## Antrenare model

După ce ai mai multe etichete cu mostre, instalează dependențele și rulează:

```bash
python -m pip install -r requirements.txt
python train_sign_model.py
```

Scriptul folosește fiecare fișier JSONL ca o secvență și evaluează clasificatorul pe 5 împărțiri ale mostrelor. Implicit, adaugă 3 variații sintetice fiecărei mostre din partea de antrenare și salvează modelul în `sign_model_augmented.joblib`. Mostrele de evaluare rămân nemodificate. Acuratețea afișată este o estimare pentru clipuri noi ale aceluiași set de date; pentru recunoașterea altor persoane trebuie colectate exemple și de la ele.

Pentru schema cu 5 mostre reale și 4 variații pentru fiecare, rulează:

```bash
python train_5x5_sign_model.py --real-per-label 5 --copies-per-real 4
```

Scriptul generează variațiile în memorie, le evaluează pe celelalte clipuri reale disponibile și salvează `sign_model_5x5.joblib`. Dacă există exact 5 mostre reale pe etichetă, poate antrena modelul, dar nu are clipuri separate pentru evaluare.

Pentru reglarea automată a setărilor clasificatorului pe datele curente, rulează:

```bash
python tune_sign_model.py
```

Acesta compară dimensiuni PCA și setări SVM prin validare în 5 părți, apoi salvează modelul ales în `sign_model_finetuned.joblib`. Programul de recunoaștere live folosește acest model implicit.

Dacă folosești mostrele mutate în `hand_dataset_vechi`, indică dosarul astfel:

```bash
python train_5x5_sign_model.py --data hand_dataset_vechi
```

## Folosirea modelului

Din dosarul `outputs`, pornește recunoașterea live cu:

```bash
python run_sign_recognizer.py
```

Apasă `SPACE`, fă un singur semn complet și apasă din nou `SPACE`. Programul afișează eticheta prezisă și nu salvează video. Apasă `Q` sau `ESC` pentru a închide. Poți verifica modelul și pe un fișier existent, fără cameră:

```bash
python run_sign_recognizer.py --clip hand_dataset/salut1.jsonl
```
