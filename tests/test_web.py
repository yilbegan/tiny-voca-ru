"""Tests for the static browser demo (player.js, worker.js, index.html, and Pyodide glue)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tiny_voca_ru import frontend
from tiny_voca_ru.config import Config
from tiny_voca_ru.frontend import stress

PUBLIC = Path(__file__).resolve().parents[1] / "web" / "public"
RUSSIAN_ONLY = "tiny-voca-ru пока читает только по-русски."

CLOCK = r"""
const fs = require("fs");
const livePlayer = new Function(fs.readFileSync(process.argv[1], "utf8") + "\nreturn livePlayer;")();
const RATE = 24000;

function fakeContext(kind = "running") {
  const context = {
    currentTime: 0, state: kind === "running" ? "running" : "suspended", destination: {}, started: [], woken: 0,
    resume() { context.woken += 1; return kind === "refused" ? Promise.reject(new Error("not allowed")) : new Promise(() => {}); },
    createBuffer(channels, length, sampleRate) {
      const data = new Float32Array(length);
      return { length, sampleRate, getChannelData: () => data };
    },
    createBufferSource() {
      const source = { buffer: null, onended: null, to: null, when: null, over: false, cut: false,
        connect(node) { source.to = node; },
        start(when) { source.when = when; context.started.push(source); },
        stop() { if (!source.over) { source.over = true; source.cut = true; if (source.onended) source.onended(); } } };
      return source;
    },
    pass(seconds) {
      context.currentTime += seconds;
      for (const source of context.started) {
        if (!source.over && source.when + source.buffer.length / source.buffer.sampleRate <= context.currentTime) {
          source.over = true;
          if (source.onended) source.onended();
        }
      }
    },
  };
  return context;
}
const tone = (length, level) => Float32Array.from({ length }, (_, i) => level * Math.sin(i / 7));
const starts = (context) => context.started.map((source) => source.when);
const facts = {};
"""

PLAYER = (
    CLOCK
    + r"""
const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
const later = () => new Promise((resolve) => setTimeout(resolve, 1));

(async () => {
{
  const context = fakeContext();
  context.currentTime = 3.25;
  let changes = 0;
  const player = livePlayer(context, () => { changes += 1; });
  const idle = player.playing;
  player.begin();
  const waiting = player.playing;
  const clips = [tone(24000, 0.1), tone(36000, 0.2), tone(12000, 0.3)];
  player.add(clips[0], RATE); context.pass(0.2);
  player.add(clips[1], RATE); context.pass(0.1);
  player.add(clips[2], RATE);
  const played = player.end();
  const before = changes;
  context.pass(2.4);
  const nearTheEnd = player.playing;
  context.pass(0.4);
  facts.early = { idle, waiting, starts: starts(context), played, nearTheEnd, after: player.playing, told: changes > before,
                  intact: context.started.every((source, i) => same(source.buffer.getChannelData(0), clips[i])),
                  rates: context.started.map((source) => source.buffer.sampleRate),
                  toTheSpeakers: context.started.every((source) => source.to === context.destination) };
}
{
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.add(tone(24000, 0.1), RATE); context.pass(1.5);
  const dry = player.playing;
  player.add(tone(24000, 0.1), RATE);
  player.add(tone(0, 0.1), RATE);
  facts.late = { starts: starts(context), dry };
}
{
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.add(tone(24000, 0.1), RATE); player.add(tone(24000, 0.1), RATE); context.pass(0.5);
  player.stop();
  const cut = context.started.map((source) => source.cut);
  const stopped = player.playing;
  player.add(tone(24000, 0.1), RATE);
  const ignored = context.started.length;
  const played = player.end();
  context.pass(4);
  player.begin();
  player.add(tone(24000, 0.1), RATE);
  facts.stop = { cut, stopped, ignored, played, again: starts(context).slice(2), playing: player.playing };
}
console.log(JSON.stringify(facts));
})();
"""
)

WORKER = r"""
const fs = require("fs");
const source = fs.readFileSync(process.argv[1], "utf8");
const take = (pattern) => source.match(pattern)[0];
const code = [take(/const ABBREVIATIONS = .*;/), take(/function sentences\(text\) \{[\s\S]*?\n\}/),
              take(/function finished\(text\) \{[\s\S]*?\n\}/), take(/function fit\(sentence\) \{[\s\S]*?\n\}/),
              take(/const PEAK = .*;/), take(/function limitPeak\(samples\) \{[\s\S]*?\n\}/),
              take(/const RUSSIAN_ONLY = .*;/), take(/async function speak\(\{[\s\S]*?\n\}/), "return speak;"].join("\n");

const wave = (text) => Float32Array.from({ length: 400 + 100 * text.length },
                                         (_, i) => (text.includes("loud") ? 1.2 : 0.5) * Math.sin(i / 5 + 1.5));
const order = [];
const messages = [];
const phonemize = (sentence) => (/[A-Za-z]/.test(sentence.replace(/XD/g, ""))
                                 ? JSON.stringify({ text: sentence, ids: [1], stress: [0] }) : "null");
const synthesize = async (phones) => { order.push("made " + phones.text); return wave(phones.text); };
const postMessage = (message, transfer = []) => {
  order.push(message.type === "sentence" ? "sent " + message.phonemes : message.type);
  messages.push(structuredClone(message, { transfer }));
};
const config = { sample_rate: 24000, model: { max_tokens: 512 } };
const speak = new Function("phonemize", "synthesize", "postMessage", "config", code)(phonemize, synthesize, postMessage, config);
const peak = (samples) => samples.reduce((most, value) => Math.max(most, Math.abs(value)), 0);

speak({ id: 7, text: process.argv[2], voice: 0, speed: 1 }).then(() => {
  const heard = messages.filter((message) => message.type === "sentence");
  const last = messages[messages.length - 1];
  const raw = heard.map((message) => wave(message.phonemes));
  const joined = Float32Array.from(raw.flatMap((samples) => Array.from(samples)));
  const scale = 0.99 / peak(joined);
  console.log(JSON.stringify({
    order,
    sentences: heard.map((message, i) => ({ id: message.id, index: message.index, total: message.total, rate: message.sampleRate,
                                            length: message.audio.length, peak: peak(message.audio),
                                            untouched: message.audio.every((value, j) => value === raw[i][j]) })),
    last: last.type !== "done" ? last :
      { type: last.type, id: last.id, rate: last.sampleRate, length: last.audio.length, peak: peak(last.audio),
        farthest: last.audio.reduce((most, value, i) => Math.max(most, Math.abs(value - joined[i] * scale)), 0) },
  }));
});
"""

STRESS = r"""
const fs = require("fs");
const source = fs.readFileSync(process.argv[1], "utf8");
const code = source.match(/async function stressUnknownWords\(text\) \{[\s\S]*?\n\}/)[0] + "\nreturn stressUnknownWords;";

class Tensor { constructor(type, data, dims) { Object.assign(this, { type, data: Array.from(data, Number), dims }); } }
const ort = { Tensor };
const asked = [];
const accent = { run: async (feeds) => {
  asked.push(Object.fromEntries(Object.entries(feeds).map(([name, tensor]) => [name, { type: tensor.type, data: tensor.data, dims: tensor.dims }])));
  const n = feeds.input_ids.dims[1];
  return { logits: { data: Float32Array.from({ length: 3 * n }, (_, i) => (i === 3 * 2 + 1 ? 5 : 0)) } };
} };
const unknownWords = () => JSON.stringify([["ой", [2, 26, 21, 3]], ["бух", [2, 13, 31, 33, 3]]]);
const learned = [];
const learnStress = (word, logits) => learned.push([word, JSON.parse(logits)]);
const stressUnknownWords = new Function("ort", "accent", "unknownWords", "learnStress", code)(ort, accent, unknownWords, learnStress);
stressUnknownWords("Ой, бух.").then(() => console.log(JSON.stringify({ asked, learned })));
"""


def glue(phonemizer: Any = frontend) -> types.SimpleNamespace:
    source = (PUBLIC / "worker.js").read_text(encoding="utf-8")
    match = re.search(
        r"^def phonemize_json\(sentence\):\n[\s\S]*?(?=^phonemize_json\()", source, re.MULTILINE
    )
    assert match is not None
    names: dict[str, Any] = {"json": json, "frontend": phonemizer}
    exec(match.group(0), names)
    return types.SimpleNamespace(
        **{
            name: names[name]
            for name in ("phonemize_json", "unknown_json", "learn_stress", "provider")
        }
    )


def node(script: str, *arguments: str) -> dict[str, Any]:
    done = subprocess.run(
        ["node", "-e", script, *arguments], capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode:
        raise AssertionError(done.stderr)
    return json.loads(done.stdout)


@unittest.skipUnless(shutil.which("node"), "needs node to run demo JS")
class LivePlayerTests(unittest.TestCase):
    facts: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.facts = node(PLAYER, str(PUBLIC / "player.js"))

    def test_sentences_chain_without_gaps(self) -> None:
        early = self.facts["early"]
        for got, want in zip(early["starts"], [3.3, 4.3, 5.8], strict=True):
            self.assertAlmostEqual(got, want, places=9)
        self.assertTrue(early["intact"] and early["toTheSpeakers"])
        self.assertFalse(early["idle"])
        self.assertTrue(
            early["waiting"] and early["played"] and early["nearTheEnd"] and early["told"]
        )
        self.assertFalse(early["after"])

    def test_late_sentence_and_stop(self) -> None:
        for got, want in zip(self.facts["late"]["starts"], [0.05, 1.55], strict=True):
            self.assertAlmostEqual(got, want, places=9)
        stop = self.facts["stop"]
        self.assertEqual(stop["cut"], [True, True])
        self.assertEqual(stop["ignored"], 2)


@unittest.skipUnless(shutil.which("node"), "needs node to run demo JS")
class WorkerTests(unittest.TestCase):
    facts: dict[str, Any]
    mixed: dict[str, Any]
    unreadable: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        worker = str(PUBLIC / "worker.js")
        cls.facts = node(WORKER, worker, "One. Two is loud. Three")
        cls.mixed = node(WORKER, worker, "XD. こんにちは. One. XD. Three.")
        cls.unreadable = {
            text: node(WORKER, worker, text) for text in ("こんにちは、世界。", "XD", "!!! ... ???")
        }

    def test_streaming_order_and_peak_limit(self) -> None:
        self.assertEqual(
            self.facts["order"],
            [
                "made One.",
                "sent One.",
                "made Two is loud.",
                "sent Two is loud.",
                "made Three.",
                "sent Three.",
                "done",
            ],
        )
        self.assertAlmostEqual(self.facts["sentences"][1]["peak"], 0.99, places=5)
        self.assertAlmostEqual(self.facts["last"]["peak"], 0.99, places=5)
        self.assertLess(self.facts["last"]["farthest"], 1e-6)

    def test_unpronounceable_sentences_skipped_and_plain_error(self) -> None:
        self.assertEqual(
            self.mixed["order"], ["made One.", "sent One.", "made Three.", "sent Three.", "done"]
        )
        for text, facts in self.unreadable.items():
            self.assertEqual(facts["order"], ["error"], text)
            self.assertEqual(
                facts["last"],
                {"type": "error", "id": 7, "message": RUSSIAN_ONLY, "plain": True},
                text,
            )

    def test_default_voices_match_config(self) -> None:
        self.assertEqual(Config().speakers, ("F", "M"))


class BrowserGlueTests(unittest.TestCase):
    def test_nothing_to_pronounce_returns_null(self) -> None:
        phonemize_json = glue().phonemize_json
        for text in ("こんにちは、世界。", "!!! ...", "\U0001f642"):
            self.assertEqual(phonemize_json(text), "null", text)
        self.assertNotEqual(phonemize_json("Добрый день, друг."), "null")

    def test_browser_tokens_match_frontend(self) -> None:
        expected = frontend.phonemize("Добрый день, как дела?")
        said = json.loads(glue().phonemize_json("Добрый день, как дела?"))
        self.assertEqual((said["ids"], said["stress"]), (list(expected.ids), list(expected.stress)))
        self.assertEqual(said["text"], frontend.transcribe("Добрый день, как дела?"))

    def test_unknown_words_stressed_via_external_onnx_accent(self) -> None:
        import numpy as np
        import onnxruntime

        browser = glue()
        model = Path(stress.__file__).with_name("data") / "accent.onnx"
        session = onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        sentence = "Флюмбрастый тряпокрут сломался."
        self.addCleanup(browser.provider.extra.clear)
        with patch.object(stress, "_accent_session", lambda: None):
            words = json.loads(browser.unknown_json(sentence))
            self.assertEqual([word for word, _ in words], ["флюмбрастый", "тряпокрут"])
            for word, ids in words:
                feed = {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                    "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64),
                }
                logits = np.asarray(session.run(["logits"], feed)[0])[0].tolist()
                browser.learn_stress(word, json.dumps(logits))
            self.assertEqual(json.loads(browser.unknown_json(sentence)), [])
            said = json.loads(browser.phonemize_json(sentence))
        first = said["stress"][1 : said["ids"].index(frontend.SYMBOL_TO_ID[frontend.SP])]
        self.assertEqual(sum(first), 1)

    @unittest.skipUnless(shutil.which("node"), "needs node to run demo JS")
    def test_worker_calls_accent_model_for_unknown_words(self) -> None:
        facts = node(STRESS, str(PUBLIC / "worker.js"))
        self.assertEqual(len(facts["asked"]), 2)
        self.assertEqual([word for word, _ in facts["learned"]], ["ой", "бух"])
