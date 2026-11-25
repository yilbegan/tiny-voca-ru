"""Gold-standard Russian G2P phonological rules, inventory, and lexicon tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tiny_voca_ru import frontend
from tiny_voca_ru.frontend.g2p import VOWEL_LETTERS, phrase_phones
from tiny_voca_ru.frontend.lexicon import ADJECTIVE_STEMS, NOUN_STEMS, WORDS
from tiny_voca_ru.frontend.stress import StressProvider
from tiny_voca_ru.frontend.symbols import (
    CONSONANTS,
    PHONE_SET,
    PHONES,
    PUNCTUATION,
    SPECIALS,
    SYMBOLS,
    VOWEL_SET,
    VOWELS,
)


def phones(text: str) -> str:
    return " | ".join(
        " ".join(("ˈ" if level else "") + phone for phone, level in word)
        for word in phrase_phones(text.split())
    )


LETTERS: dict[str, str] = {
    "д+ом": "d ˈo m",
    "д+ым": "d ˈɨ m",
    "м+ать": "m ˈa tʲ",
    "м+ять": "mʲ ˈa tʲ",
    "л+ук": "l ˈu k",
    "л+юк": "lʲ ˈu k",
    "н+ос": "n ˈo s",
    "нёс": "nʲ ˈo s",
    "б+ыл": "b ˈɨ l",
    "б+ил": "bʲ ˈi l",
    "в+ес": "vʲ ˈe s",
    "р+ад": "r ˈa t",
    "р+яд": "rʲ ˈa t",
    "п+ыль": "p ˈɨ lʲ",
    "кин+о": "kʲ i n ˈo",
    "г+ид": "ɡʲ ˈi t",
    "х+имия": "xʲ ˈi mʲ i j ə",
    "к+от": "k ˈo t",
    "к+ит": "kʲ ˈi t",
    "ст+ол": "s t ˈo l",
    "ст+ул": "s t ˈu l",
    "дв+ерь": "d vʲ ˈe rʲ",
    "л+ес": "lʲ ˈe s",
    "с+ад": "s ˈa t",
    "м+ир": "mʲ ˈi r",
    "ф+акт": "f ˈa k t",
    "ф+ильм": "fʲ ˈi lʲ m",
    "хл+еб": "x lʲ ˈe p",
    "рук+а": "r u k ˈa",
    "ног+а": "n ɐ ɡ ˈa",
    "л+ампа": "l ˈa m p ə",
    "т+уча": "t ˈu tɕ ə",
    "р+оща": "r ˈo ɕ ə",
    "к+ухня": "k ˈu x nʲ ə",
    "дер+евня": "dʲ i rʲ ˈe v nʲ ə",
    "р+ыба": "r ˈɨ b ə",
    "с+ыр": "s ˈɨ r",
    "с+ила": "sʲ ˈi l ə",
    "+ухо": "ˈu x ə",
    "+утро": "ˈu t r ə",
    "+эхо": "ˈe x ə",
    "+это": "ˈe t ə",
    "м+эр": "m ˈe r",
    "+ива": "ˈi v ə",
    "ж+ук": "ʐ ˈu k",
    "ш+ум": "ʂ ˈu m",
    "ц+ирк": "ts ˈɨ r k",
    "ж+ир": "ʐ ˈɨ r",
    "ш+ило": "ʂ ˈɨ l ə",
    "ж+есть": "ʐ ˈe sʲ tʲ",
    "ш+есть": "ʂ ˈe sʲ tʲ",
    "ц+ех": "ts ˈe x",
    "шёпот": "ʂ ˈo p ə t",
    "жёлтый": "ʐ ˈo l t ɨ j",
    "р+ожь": "r ˈo ʂ",
    "м+ышь": "m ˈɨ ʂ",
    "н+очь": "n ˈo tɕ",
    "в+ещь": "vʲ ˈe ɕ",
    "ч+ай": "tɕ ˈa j",
    "щ+ука": "ɕ ˈu k ə",
    "ч+удо": "tɕ ˈu d ə",
    "ч+ас": "tɕ ˈa s",
    "щ+и": "ɕ ˈi",
    "щ+ит": "ɕ ˈi t",
    "й+од": "j ˈo t",
    "м+ой": "m ˈo j",
    "р+ай": "r ˈa j",
    "ш+уба": "ʂ ˈu b ə",
    "жёлудь": "ʐ ˈo l u tʲ",
    "шёлк": "ʂ ˈo l k",
    "жюр+и": "ʐ u rʲ ˈi",
    "параш+ют": "p ə r ɐ ʂ ˈu t",
    "брош+юра": "b r ɐ ʂ ˈu r ə",
    "ш+ина": "ʂ ˈɨ n ə",
    "ж+изнь": "ʐ ˈɨ zʲ nʲ",
    "+яма": "j ˈa m ə",
    "+ель": "j ˈe lʲ",
    "ёж": "j ˈo ʂ",
    "+юг": "j ˈu k",
    "+ясно": "j ˈa s n ə",
    "+яблоко": "j ˈa b l ə k ə",
    "мо+я": "m ɐ j ˈa",
    "тво+я": "t v ɐ j ˈa",
    "по+ют": "p ɐ j ˈu t",
    "сто+ят": "s t ɐ j ˈa t",
    "ма+як": "m ɐ j ˈa k",
    "чит+аю": "tɕ i t ˈa j u",
    "семь+я": "sʲ i mʲ j ˈa",
    "вь+юга": "vʲ j ˈu ɡ ə",
    "подъ+езд": "p ɐ d j ˈe s t",
    "объём": "ɐ b j ˈo m",
    "съ+ел": "s j ˈe l",
    "воробь+и": "v ə r ɐ bʲ j ˈi",
    "чь+и": "tɕ j ˈi",
    "ружьё": "r u ʐ j ˈo",
    "буль+он": "b u lʲ j ˈo n",
    "компань+он": "k ə m p ɐ nʲ j ˈo n",
    "пь+еса": "pʲ j ˈe s ə",
    "стать+я": "s t ɐ tʲ j ˈa",
    "л+истья": "lʲ ˈi sʲ tʲ j ə",
    "р+ужья": "r ˈu ʐ j ə",
    "+еду": "j ˈe d u",
    "ёлка": "j ˈo l k ə",
    "ещё": "j i ɕ ˈo",
    "яз+ык": "j i z ˈɨ k",
    "яйц+о": "j i j ts ˈo",
    "+юность": "j ˈu n ə sʲ tʲ",
    "по+эт": "p ɐ ˈe t",
    "ду+эт": "d u ˈe t",
    "мо+и": "m ɐ ˈi",
    "+яблоня": "j ˈa b l ə nʲ ə",
}

REDUCTION: dict[str, str] = {
    "молок+о": "m ə l ɐ k ˈo",
    "голов+а": "ɡ ə l ɐ v ˈa",
    "хорош+о": "x ə r ɐ ʂ ˈo",
    "окн+о": "ɐ k n ˈo",
    "огор+од": "ɐ ɡ ɐ r ˈo t",
    "облак+а": "ɐ b l ɐ k ˈa",
    "автомоб+иль": "ɐ f t ə m ɐ bʲ ˈi lʲ",
    "с+ахар": "s ˈa x ə r",
    "м+ама": "m ˈa m ə",
    "дор+ога": "d ɐ r ˈo ɡ ə",
    "г+ород": "ɡ ˈo r ə t",
    "город+а": "ɡ ə r ɐ d ˈa",
    "садов+од": "s ə d ɐ v ˈo t",
    "каранд+аш": "k ə r ɐ n d ˈa ʂ",
    "паров+оз": "p ə r ɐ v ˈo s",
    "самолёт": "s ə m ɐ lʲ ˈo t",
    "трав+а": "t r ɐ v ˈa",
    "вод+а": "v ɐ d ˈa",
    "доброт+а": "d ə b r ɐ t ˈa",
    "потол+ок": "p ə t ɐ l ˈo k",
    "к+омната": "k ˈo m n ə t ə",
    "р+адуга": "r ˈa d u ɡ ə",
    "р+адость": "r ˈa d ə sʲ tʲ",
    "+облако": "ˈo b l ə k ə",
    "+азбука": "ˈa z b u k ə",
    "арб+уз": "ɐ r b ˈu s",
    "огур+ец": "ɐ ɡ u rʲ ˈe ts",
    "отвеч+ать": "ɐ t vʲ i tɕ ˈa tʲ",
    "наобор+от": "n ə ɐ b ɐ r ˈo t",
    "со+автор": "s ɐ ˈa f t ə r",
    "з+олото": "z ˈo l ə t ə",
    "полос+а": "p ə l ɐ s ˈa",
    "сторон+а": "s t ə r ɐ n ˈa",
    "раб+ота": "r ɐ b ˈo t ə",
    "раб+отать": "r ɐ b ˈo t ə tʲ",
    "разгов+ор": "r ə z ɡ ɐ v ˈo r",
    "уд+обно": "u d ˈo b n ə",
    "т+ыква": "t ˈɨ k v ə",
    "дым+ок": "d ɨ m ˈo k",
    "кур+ить": "k u rʲ ˈi tʲ",
    "был+а": "b ɨ l ˈa",
    "сыновь+я": "s ɨ n ɐ vʲ j ˈa",
    "остан+овка": "ɐ s t ɐ n ˈo f k ə",
    "н+овая": "n ˈo v ə j ə",
    "ст+арый": "s t ˈa r ɨ j",
    "эпиз+од": "e pʲ i z ˈo t",
    "жен+а": "ʐ ɨ n ˈa",
    "цен+а": "ts ɨ n ˈa",
    "шест+ой": "ʂ ɨ s t ˈo j",
    "жел+ать": "ʐ ɨ l ˈa tʲ",
    "жест+окий": "ʐ ɨ s t ˈo kʲ i j",
    "шаг+ать": "ʂ ɐ ɡ ˈa tʲ",
    "жар+а": "ʐ ɐ r ˈa",
    "к+ожа": "k ˈo ʐ ə",
    "шар+ы": "ʂ ɐ r ˈɨ",
    "шокол+ад": "ʂ ə k ɐ l ˈa t",
    "шофёр": "ʂ ɐ fʲ ˈo r",
    "цар+ица": "ts ɐ rʲ ˈi ts ə",
    "цыплёнок": "ts ɨ p lʲ ˈo n ə k",
    "жив+от": "ʐ ɨ v ˈo t",
    "душ+а": "d u ʂ ˈa",
    "к+ажется": "k ˈa ʐ ɨ ts ə",
    "в+ышел": "v ˈɨ ʂ ɨ l",
    "пошёл": "p ɐ ʂ ˈo l",
    "кон+ец": "k ɐ nʲ ˈe ts",
    "отц+а": "ɐ ts ˈa",
    "+улица": "ˈu lʲ i ts ə",
    "лес+а": "lʲ i s ˈa",
    "пят+ак": "pʲ i t ˈa k",
    "час+ы": "tɕ i s ˈɨ",
    "щек+а": "ɕ i k ˈa",
    "телеф+он": "tʲ i lʲ i f ˈo n",
    "земл+я": "zʲ i m lʲ ˈa",
    "пятёрка": "pʲ i tʲ ˈo r k ə",
    "мяч+и": "mʲ i tɕ ˈi",
    "сел+о": "sʲ i l ˈo",
    "рек+а": "rʲ i k ˈa",
    "весн+а": "vʲ i s n ˈa",
    "берёза": "bʲ i rʲ ˈo z ə",
    "дел+ить": "dʲ i lʲ ˈi tʲ",
    "тяжёлый": "tʲ i ʐ ˈo l ɨ j",
    "пятач+ок": "pʲ i t ɐ tɕ ˈo k",
    "перех+од": "pʲ i rʲ i x ˈo t",
    "чемод+ан": "tɕ i m ɐ d ˈa n",
    "часов+ой": "tɕ i s ɐ v ˈo j",
    "щав+ель": "ɕ i vʲ ˈe lʲ",
    "тётя": "tʲ ˈo tʲ ə",
    "б+уря": "b ˈu rʲ ə",
    "д+ыня": "d ˈɨ nʲ ə",
    "нед+еля": "nʲ i dʲ ˈe lʲ ə",
    "в+ишня": "vʲ ˈi ʂ nʲ ə",
    "т+ысяча": "t ˈɨ sʲ i tɕ ə",
    "в+ишням": "vʲ ˈi ʂ nʲ ə m",
    "встр+еча": "f s t rʲ ˈe tɕ ə",
    "п+амять": "p ˈa mʲ i tʲ",
    "в+ынес": "v ˈɨ nʲ i s",
    "+озеро": "ˈo zʲ i r ə",
    "+утренний": "ˈu t rʲ i nʲ i j",
    "с+иний": "sʲ ˈi nʲ i j",
    "с+иняя": "sʲ ˈi nʲ i j ə",
    "вр+емя": "v rʲ ˈe mʲ ə",
    "+имя": "ˈi mʲ ə",
    "м+ылся": "m ˈɨ l sʲ ə",
    "тётин": "tʲ ˈo tʲ i n",
}

VOICING: dict[str, str] = {
    "др+уг": "d r ˈu k",
    "гл+аз": "ɡ l ˈa s",
    "н+ож": "n ˈo ʂ",
    "лёд": "lʲ ˈo t",
    "д+уб": "d ˈu p",
    "з+уб": "z ˈu p",
    "кр+овь": "k r ˈo fʲ",
    "люб+овь": "lʲ u b ˈo fʲ",
    "м+озг": "m ˈo s k",
    "гв+оздь": "ɡ v ˈo sʲ tʲ",
    "л+одка": "l ˈo t k ə",
    "л+ожка": "l ˈo ʂ k ə",
    "пр+осьба": "p r ˈo zʲ b ə",
    "вокз+ал": "v ɐ ɡ z ˈa l",
    "футб+ол": "f u d b ˈo l",
    "вх+од": "f x ˈo t",
    "+автор": "ˈa f t ə r",
    "всё": "f sʲ ˈo",
    "тв+ой": "t v ˈo j",
    "св+ой": "s v ˈo j",
    "св+ет": "s vʲ ˈe t",
    "кварт+ира": "k v ɐ r tʲ ˈi r ə",
    "вдр+уг": "v d r ˈu k",
    "вперёд": "f pʲ i rʲ ˈo t",
    "+отдых": "ˈo d ɨ x",
    "отд+ать": "ɐ d ˈa tʲ",
    "ск+азка": "s k ˈa s k ə",
    "р+ыбка": "r ˈɨ p k ə",
    "з+ябкий": "zʲ ˈa p kʲ i j",
    "кр+уг": "k r ˈu k",
    "пир+ог": "pʲ i r ˈo k",
    "гр+иб": "ɡ rʲ ˈi p",
    "сб+ор": "z b ˈo r",
    "сд+ача": "z d ˈa tɕ ə",
    "отб+ор": "ɐ d b ˈo r",
    "берёзка": "bʲ i rʲ ˈo s k ə",
    "кн+ижка": "k nʲ ˈi ʂ k ə",
    "петр+ов": "pʲ i t r ˈo f",
    "+остров": "ˈo s t r ə f",
    "+отзыв": "ˈo d z ɨ f",
    "св+адьба": "s v ˈa dʲ b ə",
    "анекд+от": "ɐ nʲ i ɡ d ˈo t",
    "рюкз+ак": "rʲ u ɡ z ˈa k",
    "в+осемь": "v ˈo sʲ i mʲ",
    "сн+ег": "s nʲ ˈe k",
    "р+ог": "r ˈo k",
    "кл+юв": "k lʲ ˈu f",
    "рук+ав": "r u k ˈa f",
    "р+ост": "r ˈo s t",
    "в+ид": "vʲ ˈi t",
}

CLITICS: dict[str, str] = {
    "на двер+и": "n ə d vʲ i rʲ ˈi",
    "в шк+олу": "f ʂ k ˈo l u",
    "с др+угом": "z d r ˈu ɡ ə m",
    "к д+ому": "ɡ d ˈo m u",
    "из с+ада": "i s s ˈa d ə",
    "в сад+у": "f s ɐ d ˈu",
    "под стол+ом": "p ə t s t ɐ l ˈo m",
    "от бр+ата": "ɐ d b r ˈa t ə",
    "над г+ородом": "n ɐ d ɡ ˈo r ə d ə m",
    "в ит+алию": "v ɨ t ˈa lʲ i j u",
    "к игр+е": "k ɨ ɡ rʲ ˈe",
    "от +имени": "ɐ t ˈɨ mʲ i nʲ i",
    "с н+им": "s nʲ ˈi m",
    "в окн+е": "v ɐ k nʲ ˈe",
    "на окн+е": "n ə ɐ k nʲ ˈe",
    "не зн+аю": "nʲ i z n ˈa j u",
    "н+е был": "nʲ ˈe b ɨ l",
    "н+а дом": "n ˈa d ə m",
    "з+а город": "z ˈa ɡ ə r ə t",
    "п+о полю": "p ˈo p ə lʲ u",
    "из-за двер+и": "i z z ə d vʲ i rʲ ˈi",
    "по-р+усски": "p ɐ r ˈu s kʲ i",
    "кт+о-то": "k t ˈo t ə",
    "чт+о-то": "ʂ t ˈo t ə",
    "чт+о-нибудь": "ʂ t ˈo nʲ i b u tʲ",
    "гд+е-нибудь": "ɡ dʲ ˈe nʲ i b u tʲ",
    "скаж+и-ка": "s k ɐ ʐ ˈɨ k ə",
    "м+ог бы": "m ˈo ɡ b ɨ",
    "+он же": "ˈo n ʐ ə",
    "зн+аешь ли": "z n ˈa j i ʂ lʲ i",
    "т+ы ли": "t ˈɨ lʲ i",
    "др+уг же": "d r ˈu ɡ ʐ ə",
    "и т+ы": "i t ˈɨ",
    "а +он": "ɐ ˈo n",
    "в +этом": "v ˈe t ə m",
    "без в+ас": "bʲ i z v ˈa s",
    "из вс+ех": "i s f sʲ ˈe x",
    "к в+ам": "k v ˈa m",
    "в дв+ух": "v d v ˈu x",
    "с в+ами": "s v ˈa mʲ i",
    "в ф+ильмах": "fʲ fʲ ˈi lʲ m ə x",
    "в в+оду": "v v ˈo d u",
    "с сестр+ой": "sʲ sʲ i s t r ˈo j",
    "без с+ил": "bʲ i sʲ sʲ ˈi l",
    "с соб+ой": "s s ɐ b ˈo j",
    "в футб+олке": "f f u d b ˈo l kʲ i",
    "к кот+орой": "k k ɐ t ˈo r ə j",
    "без сп+утников": "bʲ i s s p ˈu t nʲ i k ə f",
    "с зап+олненными": "z z ɐ p ˈo l nʲ i n ɨ mʲ i",
    "до свид+ания": "d ə s vʲ i d ˈa nʲ i j ə",
    "по дор+огам": "p ə d ɐ r ˈo ɡ ə m",
    "но д+ом": "n o d ˈo m",
    "то л+и": "t o lʲ ˈi",
    "что д+ом": "ʂ t o d ˈo m",
    "н+а дв+ор в+ышел м+альчик": "n ˈa | d v ˈo r | v ˈɨ ʂ ɨ l | m ˈa lʲ tɕ i k",
    "с+ад з+елен": "s ˈa t | zʲ ˈe lʲ i n",
    "в+от д+ом": "v ˈo t | d ˈo m",
    "я зн+аю": "j i z n ˈa j u",
}

CLUSTERS: dict[str, str] = {
    "ч+естный": "tɕ ˈe s n ɨ j",
    "м+естный": "mʲ ˈe s n ɨ j",
    "изв+естно": "i z vʲ ˈe s n ə",
    "п+оздно": "p ˈo z n ə",
    "звёздный": "z vʲ ˈo z n ɨ j",
    "с+олнца": "s ˈo n ts ə",
    "ч+увство": "tɕ ˈu s t v ə",
    "ч+увствовать": "tɕ ˈu s t v ə v ə tʲ",
    "здр+авствуй": "z d r ˈa s t v u j",
    "сч+астлив": "ɕ ˈa s lʲ i f",
    "счастл+ивый": "ɕ i s lʲ ˈi v ɨ j",
    "с+ердца": "sʲ ˈe r ts ə",
    "гиг+антский": "ɡʲ i ɡ ˈa n s kʲ i j",
    "голл+андский": "ɡ ɐ l ˈa n s kʲ i j",
    "тур+истский": "t u rʲ ˈi s kʲ i j",
    "рентг+ен": "rʲ i n ɡʲ ˈe n",
    "ландш+афт": "l ɐ n ʂ ˈa f t",
    "уч+иться": "u tɕ ˈi ts ə",
    "+учится": "ˈu tɕ i ts ə",
    "смеётся": "s mʲ i j ˈo ts ə",
    "заним+аться": "z ə nʲ i m ˈa ts ə",
    "кат+аться": "k ɐ t ˈa ts ə",
    "д+етский": "dʲ ˈe ts kʲ i j",
    "городск+ой": "ɡ ə r ɐ ts k ˈo j",
    "сов+етский": "s ɐ vʲ ˈe ts kʲ i j",
    "бр+атский": "b r ˈa ts kʲ i j",
    "д+етство": "dʲ ˈe ts t v ə",
    "сос+едство": "s ɐ sʲ ˈe ts t v ə",
    "отв+етственный": "ɐ t vʲ ˈe ts t vʲ i n ɨ j",
    "дв+адцать": "d v ˈa ts ə tʲ",
    "тр+идцать": "t rʲ ˈi ts ə tʲ",
    "лётчик": "lʲ ˈo tɕ i k",
    "+отчество": "ˈo tɕ i s t v ə",
    "перев+одчик": "pʲ i rʲ i v ˈo tɕ i k",
    "отчёт": "ɐ tɕ ˈo t",
    "счёт": "ɕ ˈo t",
    "счит+ать": "ɕ i t ˈa tʲ",
    "мужч+ина": "m u ɕ ˈi n ə",
    "изв+озчик": "i z v ˈo ɕ i k",
    "расчёт": "r ɐ ɕ ˈo t",
    "сш+ить": "ʂ ˈɨ tʲ",
    "сж+ечь": "ʐ ˈe tɕ",
    "в+ысший": "v ˈɨ ʂ ɨ j",
    "н+изший": "nʲ ˈi ʂ ɨ j",
    "бесш+умный": "bʲ i ʂ ˈu m n ɨ j",
    "+езжу": "j ˈe ʐ u",
    "приезж+ать": "p rʲ i j i ʐ ˈa tʲ",
    "разж+ать": "r ɐ ʐ ˈa tʲ",
    "сж+ать": "ʐ ˈa tʲ",
    "без ш+ума": "bʲ i ʂ ʂ ˈu m ə",
    "с ш+умом": "ʂ ʂ ˈu m ə m",
    "с ж+аром": "ʐ ʐ ˈa r ə m",
    "с ч+ем": "ɕ tɕ ˈe m",
    "из ч+ашки": "i ɕ tɕ ˈa ʂ kʲ i",
    "от ц+ели": "ɐ t ts ˈe lʲ i",
    "легк+о": "lʲ i x k ˈo",
    "м+ягкий": "mʲ ˈa x kʲ i j",
    "лёгкий": "lʲ ˈo x kʲ i j",
    "м+ягко": "mʲ ˈa x k ə",
    "иск+усство": "i s k ˈu s t v ə",
    "к+асса": "k ˈa s ə",
    "в+анна": "v ˈa n ə",
    "гр+уппа": "ɡ r ˈu p ə",
    "с+умма": "s ˈu m ə",
    "дл+инный": "d lʲ ˈi n ɨ j",
    "сс+ора": "s ˈo r ə",
    "отт+уда": "ɐ t ˈu d ə",
    "расск+аз": "r ɐ s k ˈa s",
    "кл+асс": "k l ˈa s",
    "бесс+ильный": "bʲ i sʲ ˈi lʲ n ɨ j",
    "вв+ерх": "vʲ ˈe r x",
    "сс+ылка": "s ˈɨ l k ə",
    "т+еннис": "t ˈe nʲ i s",
    "+есть": "j ˈe sʲ tʲ",
    "г+ость": "ɡ ˈo sʲ tʲ",
    "к+ость": "k ˈo sʲ tʲ",
    "п+есня": "pʲ ˈe sʲ nʲ ə",
    "м+остик": "m ˈo sʲ tʲ i k",
    "к+ончик": "k ˈo nʲ tɕ i k",
    "ж+енщина": "ʐ ˈe nʲ ɕ i n ə",
    "звезд+е": "z vʲ i zʲ dʲ ˈe",
    "гв+озди": "ɡ v ˈo zʲ dʲ i",
    "зд+есь": "z dʲ ˈe sʲ",
    "стен+а": "s tʲ i n ˈa",
}

LEXICON: dict[str, str] = {
    "что": "ʂ t o",
    "чт+о": "ʂ t ˈo",
    "чт+обы": "ʂ t ˈo b ɨ",
    "ничт+о": "nʲ i ʂ t ˈo",
    "кон+ечно": "k ɐ nʲ ˈe ʂ n ə",
    "кон+ечный": "k ɐ nʲ ˈe tɕ n ɨ j",
    "ск+учно": "s k ˈu ʂ n ə",
    "ск+учный": "s k ˈu ʂ n ɨ j",
    "нар+очно": "n ɐ r ˈo ʂ n ə",
    "я+ичница": "j i ˈi ʂ nʲ i ts ə",
    "сег+одня": "sʲ i v ˈo d nʲ ə",
    "сег+одняшний": "sʲ i v ˈo d nʲ i ʂ nʲ i j",
    "б+ог": "b ˈo x",
    "б+ога": "b ˈo ɡ ə",
    "ег+о": "j i v ˈo",
    "н+ового": "n ˈo v ə v ə",
    "втор+ого": "f t ɐ r ˈo v ə",
    "моег+о": "m ə j i v ˈo",
    "всег+о": "f sʲ i v ˈo",
    "ничег+о": "nʲ i tɕ i v ˈo",
    "ког+о": "k ɐ v ˈo",
    "тог+о": "t ɐ v ˈo",
    "одног+о": "ɐ d n ɐ v ˈo",
    "кр+асного": "k r ˈa s n ə v ə",
    "с+инего": "sʲ ˈi nʲ i v ə",
    "мн+ого": "m n ˈo ɡ ə",
    "немн+ого": "nʲ i m n ˈo ɡ ə",
    "стр+ого": "s t r ˈo ɡ ə",
    "д+орого": "d ˈo r ə ɡ ə",
    "дорог+ого": "d ə r ɐ ɡ ˈo v ə",
    "т+ест": "t ˈe s t",
    "т+есты": "t ˈe s t ɨ",
    "т+есто": "tʲ ˈe s t ə",
    "мод+ель": "m ɐ d ˈe lʲ",
    "от+ель": "ɐ t ˈe lʲ",
    "каф+е": "k ɐ f ˈe",
    "шосс+е": "ʂ ɐ s ˈe",
    "т+емп": "t ˈe m p",
    "ст+енд": "s t ˈe n t",
    "пюр+е": "pʲ u r ˈe",
    "куп+е": "k u p ˈe",
    "тир+е": "tʲ i r ˈe",
    "бр+енд": "b r ˈe n t",
    "р+ейтинг": "r ˈe j tʲ i n k",
    "ф+ейк": "f ˈe j k",
    "к+етчуп": "k ˈe tɕ u p",
    "интерн+ет": "i n t e r n ˈe t",
    "тестиров+ания": "t e s tʲ i r ɐ v ˈa nʲ i j ə",
    "м+енеджер": "m ˈe n e d ʐ e r",
    "компь+ютер": "k ɐ m pʲ j ˈu t e r",
    "детект+ив": "d e t e k tʲ ˈi f",
    "б+изнес": "bʲ ˈi z n e s",
    "интерф+ейс": "i n t e r f ˈe j s",
    "тест+ировщик": "t e s tʲ ˈi r ə f ɕ i k",
    "р+адио": "r ˈa dʲ i o",
    "как+ао": "k ɐ k ˈa o",
    "тр+ио": "t rʲ ˈi o",
}

GOLD: dict[str, str] = {**LETTERS, **REDUCTION, **VOICING, **CLITICS, **CLUSTERS, **LEXICON}


class GoldStandardTests(unittest.TestCase):
    def check(self, table: dict[str, str]) -> None:
        wrong = {
            text: (phones(text), expected)
            for text, expected in table.items()
            if phones(text) != expected
        }
        self.assertEqual(wrong, {}, "text: (got, expected)")

    def test_letters_softness_iotation(self) -> None:
        self.check(LETTERS)

    def test_vowel_reduction(self) -> None:
        self.check(REDUCTION)

    def test_voicing(self) -> None:
        self.check(VOICING)

    def test_clitics_join_their_host(self) -> None:
        self.check(CLITICS)

    def test_clusters_merges_geminates(self) -> None:
        self.check(CLUSTERS)

    def test_lexicon(self) -> None:
        self.check(LEXICON)

    def test_post_tonic_final_e_minimal_pairs(self) -> None:
        self.assertEqual(phones("в+аше"), "v ˈa ʂ ə")
        self.assertEqual(phones("в+аши"), "v ˈa ʂ ɨ")
        self.assertNotEqual(phones("в+аше"), phones("в+аши"))
        self.assertNotEqual(phones("сл+едующее"), phones("сл+едующие"))
        self.assertNotEqual(phones("стол+етнее"), phones("стол+етние"))
        self.assertNotEqual(phones("серьёзнейшее"), phones("серьёзнейшие"))
        self.assertNotEqual(phones("ц+енные"), phones("ц+енное"))
        self.assertNotEqual(phones("нал+ичие"), phones("нал+ичия"))
        self.assertEqual(phones("б+ольше"), "b ˈo lʲ ʂ ə")
        self.assertEqual(phones("с+олнце"), "s ˈo n ts ə")
        self.assertEqual(phones("м+оре"), "m ˈo rʲ i")
        self.assertEqual(phones("в д+оме"), "v d ˈo mʲ i")
        self.assertEqual(phones("нал+ичие"), "n ɐ lʲ ˈi tɕ i j i")
        self.assertEqual(phones("ц+енные"), "ts ˈe n ɨ j i")
        self.assertEqual(phones("в+ынес"), "v ˈɨ nʲ i s")
        self.assertEqual(phones("н+ижнем"), "nʲ ˈi ʐ nʲ i m")

    def test_table_is_large(self) -> None:
        self.assertGreater(len(GOLD), 350)


class InventoryTests(unittest.TestCase):
    def test_inventory(self) -> None:
        self.assertEqual(
            (len(SPECIALS), len(PUNCTUATION), len(CONSONANTS), len(VOWELS)), (5, 12, 36, 8)
        )
        self.assertEqual(len(SYMBOLS), 61)
        self.assertEqual(len(set(SYMBOLS)), 61)
        self.assertEqual(SYMBOLS[:5], ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>"))
        self.assertEqual(PHONES[-8:], ("a", "o", "e", "i", "u", "ɨ", "ɐ", "ə"))

    def test_every_output_phone_is_in_the_inventory(self) -> None:
        texts = [*GOLD, "ъ ь", "abc д+ом", "-то", "в", "Ёлка-палка", "п+одъ-езд", "++а", "+"]
        seen: set[str] = set()
        for text in texts:
            for word in phrase_phones(text.split()):
                self.assertTrue(word, text)
                for phone, level in word:
                    self.assertIn(phone, PHONE_SET, text)
                    self.assertIn(level, (0, 1))
                    if level:
                        self.assertIn(phone, VOWEL_SET, f"stress on a consonant in {text}")
                    seen.add(phone)
        self.assertEqual(seen, PHONE_SET, "the gold table should use every phone")

    def test_one_stressed_vowel_per_phonetic_word(self) -> None:
        for text in GOLD:
            for word in phrase_phones(text.split()):
                if any(level for _, level in word):
                    self.assertEqual(sum(level for _, level in word), 1, text)

    def test_respellings_keep_the_vowels(self) -> None:
        for table in (WORDS, NOUN_STEMS, ADJECTIVE_STEMS):
            for word, respelling in table.items():
                self.assertEqual(
                    sum(c in VOWEL_LETTERS for c in word),
                    sum(c in VOWEL_LETTERS for c in respelling),
                    word,
                )


class Refuses(StressProvider):
    name = "refuses"

    def accentuate(self, text: str) -> str:
        raise AssertionError("stressed text must not go to the stress provider")


class Marks(StressProvider):
    name = "marks"

    def accentuate(self, text: str) -> str:
        return text.replace("Дом", "Д+ом").replace("дом", "д+ом").replace("стоит", "сто+ит")


class PiecesTests(unittest.TestCase):
    def test_stressed_text_needs_no_provider(self) -> None:
        with patch(
            "tiny_voca_ru.frontend.tokenize.stress_provider",
            side_effect=AssertionError("unexpected"),
        ):
            result = frontend.pieces("Н+а дв+ор в+ышел м+альчик.", stressed=True)
            self.assertEqual(
                frontend.transcribe("Н+а дв+ор в+ышел м+альчик.", stressed=True),
                "nˈa dvˈor vˈɨʂɨl mˈalʲtɕik .",
            )
        self.assertEqual([value for punctuation, value in result if punctuation], ["."])
        [(_, words)] = [piece for piece in result if not piece[0]]
        self.assertEqual(len(words), 4)
        with self.assertRaises(ValueError):
            frontend.pieces("Д+ом.", stress=Refuses(), stressed=True)

    def test_stressed_text_is_only_cleaned(self) -> None:
        self.assertEqual(
            frontend.transcribe("«Ну чт+о» -- д+ом?..", stressed=True), '" nuʂtˈo " — dˈom ?'
        )
        for text in ("в 2 ч+аса", "м+ой iPhone", "р+ост 5%"):
            with self.subTest(text=text), self.assertRaises(frontend.NotNormalized):
                frontend.pieces(text, stressed=True)

    def test_plain_text_goes_to_the_provider(self) -> None:
        self.assertEqual(frontend.transcribe("Дом стоит.", stress=Marks()), "dˈom stɐˈit .")

    def test_one_mark_does_not_switch_the_provider_off(self) -> None:
        self.assertEqual(frontend.transcribe("Д+ом стоит.", stress=Marks()), "dˈom stɐˈit .")

    def test_punctuation_separates_phrases(self) -> None:
        result = frontend.pieces("Кт+о-то пришёл, а т+ы н+ет!", stressed=True)
        self.assertEqual([value for punctuation, value in result if punctuation], [",", "!"])
        self.assertEqual([len(value) for punctuation, value in result if not punctuation], [2, 2])

    def test_hyphenated_particles_are_clitics(self) -> None:
        for word, expected in (
            ("чт+о-то", "ʂ t ˈo t ə"),
            ("кт+о-либо", "k t ˈo lʲ i b ə"),
            ("когд+а-нибудь", "k ɐ ɡ d ˈa nʲ i b u tʲ"),
            ("дав+ай-ка", "d ɐ v ˈa j k ə"),
            ("всё-таки", "f sʲ ˈo t ə kʲ i"),
            ("кое-кт+о", "k ə j i k t ˈo"),
            ("из-за дв+ери", "i z z ɐ d vʲ ˈe rʲ i"),
            ("по-р+усски", "p ɐ r ˈu s kʲ i"),
        ):
            with self.subTest(word=word):
                self.assertEqual(phones(word), expected)
        self.assertEqual(phones("чт+о т+о"), "ʂ t ˈo | t ˈo")
