/* Browser-side inference over recorded Snake states. */
(function () {
  'use strict';

  let model;
  let frameIndex = -1;

  function summarize(data) {
    const backgroundEdges = [];
    const layerStats = data.weights.map(function (matrix, layer) {
      matrix.forEach(function (row, target) {
        const strongest = row.map((w, s) => ({ w, s }))
          .sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 2);
        strongest.forEach(edge => backgroundEdges.push({ f: layer, t: target, ...edge }));
      });
      const magnitudes = matrix.flat().map(Math.abs);
      const biases = data.biases[layer + 1];
      return {
        mean_abs: mean(magnitudes),
        max_abs: Math.max(...magnitudes),
        bias_mean: mean(biases),
        bias_abs_mean: mean(biases.map(Math.abs)),
        delta: null,
      };
    });
    return { ...data, backgroundEdges, layerStats };
  }

  function forward(features, weights, biases) {
    const activations = [features];
    const zs = [null];
    weights.forEach(function (matrix, layer) {
      const inputs = activations[layer];
      const values = matrix.map(function (row, target) {
        return row.reduce(function (sum, weight, source) {
          return sum + weight * inputs[source];
        }, biases[layer + 1][target]);
      });
      zs.push(values);
      activations.push(layer === weights.length - 1
        ? values : values.map(value => Math.max(0, value)));
    });
    return { activations, zs };
  }

  function transition(frame, q, action, weights, biases, learning, names) {
    if (frame.reward === undefined) return null;
    const next = forward(frame.next_features, weights, biases).activations.at(-1);
    const bestFuture = Math.max(...next);
    const target = frame.reward + (frame.done ? 0 : learning.gamma * bestFuture);
    return {
      action_name: names[action],
      reward: frame.reward,
      event: frame.event,
      done: frame.done,
      predicted: q[action],
      best_future: bestFuture,
      target,
      td_error: target - q[action],
      squared_error: (target - q[action]) ** 2,
    };
  }

  async function payload(selection, scenario, advance) {
    if (!model) {
      model = fetch('./demo.json').then(response => {
        if (!response.ok) throw new Error('Could not load the demo');
        return response.json();
      }).then(summarize).catch(error => {
        model = null;
        throw error;
      });
    }
    const data = await model;
    const { weights, biases, frames, history, learning, layer_labels } = data;
    if (scenario === 'auto' && (advance || frameIndex === -1)) {
      frameIndex = (frameIndex + 1) % frames.length;
    }
    const frame = scenario === 'auto' ? frames[frameIndex] : data.scenarios[scenario];
    const { layer, neuron } = selection;
    const { activations, zs } = forward(frame.features, weights, biases);
    const q = activations.at(-1);
    const action = q.indexOf(Math.max(...q));
    const row = weights[layer][neuron];
    const inputs = activations[layer];
    const stats = history.at(-1);
    const edges = data.backgroundEdges.filter(edge =>
      !(edge.f === layer && edge.t === neuron) &&
      !(edge.f === layer + 1 && edge.s === neuron)
    );
    const scenarioLabel = {
      auto: 'Recorded game replay',
      start: 'Game start',
      mid: 'Mid-game (scripted)',
    };
    return {
      ok: true,
      version: ['replay', 0, 'static'],
      all_weights: weights,
      episode: stats.episode,
      total_episodes: stats.episode,
      stats,
      sizes: data.sizes,
      layer_labels,
      action_names: data.action_names,
      parameter_counts: weights.map(matrix => matrix.length * (matrix[0].length + 1)),
      scenario,
      scenario_label: scenarioLabel[scenario],
      board: { ...frame, action, q },
      activations,
      zs,
      biases,
      active: {
        h1: activations[1].filter(value => value > 0).length,
        h2: activations[2].filter(value => value > 0).length,
        output: action,
      },
      chosen_action: action,
      selected: {
        layer,
        neuron,
        layer_label: layer_labels[layer + 1],
        input_label: layer_labels[layer],
        is_output: layer === 2,
        bias: biases[layer + 1][neuron],
        weight_norm: Math.hypot(...row),
        weights: row,
        input_names: layer === 0 ? window.NetworkView.featureNames
          : inputs.map((_, index) => 'h' + layer + '[' + index + ']'),
        inputs,
        contributions: row.map((weight, source) => weight * inputs[source]),
        z: zs[layer + 1][neuron],
        activation: activations[layer + 1][neuron],
      },
      edges,
      focus_in: row.map((w, s) => ({ w, s })),
      focus_out: (weights[layer + 1] || []).map((incoming, t) => ({ w: incoming[neuron], t })),
      layer_stats: data.layerStats,
      history: [],
      layer_history: [],
      training_history: history,
      learning: {
        ...learning,
        epsilon: stats.epsilon,
        average_loss: stats.average_loss,
        demo: transition(frame, q, action, weights, biases, learning, data.action_names),
      },
    };
  }

  function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  window.SnakeReplay = { payload };
})();
