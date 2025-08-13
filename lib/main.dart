import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AviatorPredictorApp());
}

class AviatorPredictorApp extends StatelessWidget {
  const AviatorPredictorApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Aviator Predictor (Local)',
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: const Color(0xFF7C4DFF)),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const _keyHistory = 'aviator_history';
  static const _keyWeights = 'aviator_weights';

  final TextEditingController _inputCtrl = TextEditingController();
  List<double> history = [];

  late final OnlineLogReg model;

  @override
  void initState() {
    super.initState();
    model = OnlineLogReg(nFeatures: featureCount);
    _load();
  }

  Future<void> _load() async {
    final sp = await SharedPreferences.getInstance();
    final rawH = sp.getString(_keyHistory);
    if (rawH != null) {
      final list = (jsonDecode(rawH) as List).map((e) => (e as num).toDouble()).toList();
      setState(() => history = list);
    }
    final rawW = sp.getString(_keyWeights);
    if (rawW != null) {
      model.fromJson(rawW);
    }
  }

  Future<void> _save() async {
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_keyHistory, jsonEncode(history));
    await sp.setString(_keyWeights, model.toJson());
  }

  void _addMultiplier() {
    final txt = _inputCtrl.text.trim().replaceAll(',', '.');
    final x = double.tryParse(txt);
    if (x == null || x <= 0) return;
    setState(() => history.add(x));
    _inputCtrl.clear();
    _save();
  }

  void _clearAll() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Effacer les données ?'),
        content: const Text('Historique et paramètres du modèle seront supprimés.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Annuler')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Effacer')),
        ],
      ),
    );
    if (ok ?? false) {
      setState(() { history.clear(); model.reset(); });
      _save();
    }
  }

  // ====== ANALYSES ======
  double? get ewma => history.isEmpty ? null : _ewma(history, alpha: 0.2);
  double? get vol  => history.length < 2 ? null : _ewmStd(history, alpha: 0.2);
  PageHinkleyResult get ph => pageHinkley(history, delta: 0.005, lambda: 0.5, alpha: 0.99);

  // Entraîne le modèle avec des étiquettes binaires pour divers seuils
  void _trainModel() {
    if (history.length < 20) return; // besoin d'un minimum
    // Fenêtres glissantes pour créer des exemples (features -> label)
    for (int i = featureWindow; i < history.length - 1; i++) {
      final feats = makeFeatures(history.sublist(0, i));
      final next = [i];
      // multi-objectif : on entraîne 4 têtes (1.5x, 2x, 3x, 5x) via partage de poids
      final targets = [1.5, 2.0, 3.0, 5.0].map((t) => next >= t ? 1.0 : 0.0).toList();
      model.fit(feats, targets);
    }
    _save();
    setState(() {});
  }

  Map<double, double> _predictNow() {
    final feats = makeFeatures(history);
    final tList = [1.5, 2.0, 3.0, 5.0];
    final probs = <double, double>{};
    for (final t in tList) {
      probs[t] = model.predict(feats, threshold: t);
    }
    return probs;
  }

  @override
  Widget build(BuildContext context) {
    final probs = history.length >= featureWindow ? _predictNow() : null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Aviator Predictor – Analyse locale'),
        actions: [
          IconButton(onPressed: _clearAll, icon: const Icon(Icons.delete_outline)),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Ajouter un multiplicateur', style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Row(children: [
                    Expanded(
                      child: TextField(
                        controller: _inputCtrl,
                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                        decoration: const InputDecoration(hintText: 'Ex: 1.23, 2.01, 7.59…'),
                        onSubmitted: (_) => _addMultiplier(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.icon(onPressed: _addMultiplier, icon: const Icon(Icons.add), label: const Text('Ajouter')),
                  ]),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          if (history.isNotEmpty)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Statistiques rapides', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    Wrap(spacing: 12, runSpacing: 12, children: [
                      _statChip('EWMA', ewma?.toStringAsFixed(3) ?? '–'),
                      _statChip('Volatilité (EW)', vol?.toStringAsFixed(3) ?? '–'),
                      _statChip('Rupture ?', ph.changeDetected ? 'Oui' : 'Non'),
                      _statChip('Taille série', history.length.toString()),
                    ]),
                  ],
                ),
              ),
            ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Modèle probabiliste (SGD)', style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  const Text('Entraînement local sur vos données (aucun envoi réseau).'),
                  const SizedBox(height: 8),
                  Row(children: [
                    FilledButton.icon(onPressed: _trainModel, icon: const Icon(Icons.school), label: const Text('Entraîner')), 
                    const SizedBox(width: 8),
                    if (history.length < 20) const Text('Ajoutez ≥ 20 points pour entraîner.', style: TextStyle(color: Colors.black54)),
                  ]),
                  const SizedBox(height: 12),
                  if (probs != null)
                    Column(
crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Probabilité que le prochain ≥ seuil :'),
                        const SizedBox(height: 8),
                        for (final entry in probs.entries)
                          ListTile(
                            leading: const Icon(Icons.bolt),
                            title: Text('Seuil ${entry.key}x'),
                            trailing: Text('${(entry.value * 100).toStringAsFixed(1)} %', style: const TextStyle(fontWeight: FontWeight.w700)),
                          ),
                        const SizedBox(height: 6),
                        const Text('⚠️ Aucune garantie : les jeux “provably fair” sont conçus pour être imprévisibles.'),
                      ],
                    )
                  else
                    const Text('Ajoutez encore quelques valeurs pour obtenir des estimations.'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          if (history.isNotEmpty)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Historique (dernier en bas)', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [for (final x in history) Chip(label: Text(x.toStringAsFixed(2))))],
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

Widget _statChip(String title, String value) {
  return Chip(
    label: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(title, style: const TextStyle(fontSize: 12, color: Colors.black54)),
      Text(value, style: const TextStyle(fontWeight: FontWeight.w700)),
    ]),
  );
}

// ===== Fonctions d'analyse =====

double _ewma(List<double> x, {required double alpha}) {
  double s = x.first;
  for (int i = 1; i < x.length; i++) {
    s = alpha * x[i] + (1 - alpha) * s;
  }
  return s;
}

double _ewmStd(List<double> x, {required double alpha}) {
  if (x.length < 2) return 0;
  double mean = x.first;
  double varE = 0; // variance exponentielle
  for (int i = 1; i < x.length; i++) {
    final prevMean = mean;
    mean = alpha * x[i] + (1 - alpha) * mean;
    varE = alpha * math.pow(x[i] - prevMean, 2).toDouble() + (1 - alpha) * varE;
  }
  return math.sqrt(varE);
}

class PageHinkleyResult { final bool changeDetected; final int? index; const PageHinkleyResult(this.changeDetected, this.index); }
PageHinkleyResult pageHinkley(List<double> x, {double delta = 0.005, double lambda = 0.5, double alpha = 0.99}) {
  if (x.isEmpty) return const PageHinkleyResult(false, null);
  double mean = x.first;
  double mT = 0; double M = 0;
  int? idx;
  for (int i = 1; i < x.length; i++) {
    mean = alpha * x[i] + (1 - alpha) * mean;
    mT += x[i] - mean - delta;
    M = math.min(M, mT);
    if (mT - M > lambda) { idx = i; break; }
  }
  return PageHinkleyResult(idx != null, idx);
}

// ===== Modèle : régression logistique en ligne (multi-seuils via biais spécifique) =====

const featureWindow = 10; // nombre de points pour fabriquer les features
const featureCount = 8;   // doit correspondre à makeFeatures

List<double> makeFeatures(List<double> hist) {
  final h = hist.take(hist.length).toList();
  if (h.length < featureWindow) {
    // padding avec la moyenne
    final padVal = h.isEmpty ? 1.0 : h.reduce((a, b) => a + b) / h.length;
    while (h.length < featureWindow) h.insert(0, padVal);
  } else if (h.length > featureWindow) {
    h.removeRange(0, h.length - featureWindow);
  }
  final mean = h.reduce((a, b) => a + b) / h.length;
  final std  = (h.map((e) => (e - mean) * (e - mean)).reduce((a, b) => a + b) / h.length);
  final sdev = math.sqrt(std + 1e-9);

  double last = h.last;
  double minv = h.reduce(math.min);
  double maxv = h.reduce(math.max);
  double range = (maxv - minv) + 1e-9;

  // Caractéristiques simples mais informatives
  final feats = <double>[
    (last - mean) / sdev,
    (maxv - mean) / sdev,
    (mean - minv) / sdev,
    (h.last - h.first) / (range),
    _ewma(h, alpha: 0.2),
    _ewmStd(h, alpha: 0.2),
    // Fréquence des gros crash ( < 1.2x )
    h.where((e) => e < 1.2).length / h.length,
    // Fréquence des gros vols ( >= 5x )
    h.where((e) => e >= 5.0).length / h.length,
  ];
  return feats;
}

class OnlineLogReg {
  final int nFeatures;
  // Poids partagés + biais par seuil
  late List<double> w; // taille nFeatures
  final Map<double, double> biases = {1.5: 0.0, 2.0: 0.0, 3.0: 0.0, 5.0: 0.0};
  double lr = 0.02;
  OnlineLogReg({required this.nFeatures}) { reset(); }

  void reset() { w = List.filled(nFeatures, 0.0); biases.updateAll((k, v) => 0.0); }

  double _dot(List<double> a, List<double> b) {
    double s = 0.0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
  }
  double _sigmoid(double z) => 1.0 / (1.0 + math.exp(-z));

  void fit(List<double> x, List<double> targets) {
    // Mise à jour conjointe des poids partagés en sommant les gradients sur les têtes
    final grads = List<double>.filled(nFeatures, 0.0);
    biases.forEach((th, b) {
      final y = targets[ {1.5:0, 2.0:1, 3.0:2, 5.0:3}[th]! ];
      final p = _sigmoid(_dot(w, x) + biases[th]!);
      final err = p - y; // gradient de log-loss
      for (int i = 0; i < nFeatures; i++) grads[i] += err * x[i];
      biases[th] = biases[th]! - lr * err; // maj biais spécifique au seuil
    });
    for (int i = 0; i < nFeatures; i++) { w[i] -= lr * grads[i]; }
  }

  double predict(List<double> x, {required double threshold}) {
    final z = _dot(w, x) + (biases[threshold] ?? 0.0);
    return _sigmoid(z);
  }

  String toJson() => jsonEncode({ 'w': w, 'b': biases, 'lr': lr });
  void fromJson(String s) {
    final m = jsonDecode(s) as Map<String, dynamic>;
    w = (m['w'] as List).map((e) => (e as num).toDouble()).toList();
    final bb = (m['b'] as Map).map((k, v) => MapEntry(double.parse(k.toString()), (v as num).toDouble()));
    biases..clear()..addAll(bb);
    lr = (m['lr'] as num).toDouble();
  }
}
