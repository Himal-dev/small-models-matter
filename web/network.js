/* Full-network SVG view: 24 -> 128 -> 128 -> 3. Responsive. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const SIZES = [24, 128, 128, 3];
  const ACTION_NAMES = ['STRAIGHT', 'TURN RIGHT', 'TURN LEFT'];
  const FEATURE_NAMES = [
    'danger straight', 'danger right', 'danger left',
    'heading right', 'heading down', 'heading left', 'heading up',
    'food left', 'food right', 'food up', 'food down',
    'wall straight', 'wall right', 'wall left',
    'body straight', 'body right', 'body left',
    'food dx', 'food dy', 'snake length', 'idle ratio',
    'same column', 'same row', 'score',
  ];

  const WIDE = {
    narrow: false,
    X: [222, 560, 860, 1140],
    viewBox: '0 0 1300 960',
    top: 80,
    bottom: 900,
    outputY: [390, 480, 570],
    inputLabels: true,
    outputLabelsRight: true,
    hiddenRadius: 2.5,
    inputRadius: 4.5,
    labelSize: 12,
    titles: [
      'INPUT \u00b7 24 features',
      'HIDDEN 1 \u00b7 128 neurons (ReLU)',
      'HIDDEN 2 \u00b7 128 neurons (ReLU)',
      'OUTPUT \u00b7 3 Q-values',
    ],
  };

  const NARROW = {
    narrow: true,
    X: [70, 330, 590, 820],
    viewBox: '0 0 900 760',
    top: 70,
    bottom: 700,
    outputY: [260, 380, 500],
    inputLabels: false,
    outputLabelsRight: false,
    hiddenRadius: 2.2,
    inputRadius: 4.0,
    labelSize: 13.5,
    titles: [
      'INPUT \u00b7 24',
      'HIDDEN 1 \u00b7 128',
      'HIDDEN 2 \u00b7 128',
      'OUTPUT \u00b7 3',
    ],
  };

  let layout = WIDE;
  let svg = null;
  let edgeGroup = null;
  let denseEdgeGroup = null;
  let denseEdgeKey = null;
  let nodeGroup = null;
  let labelGroup = null;
  let tooltip = null;
  let selectionRing = null;
  let nodes = [[], [], [], []];
  let inputValueEls = [];
  let inputNameEls = [];
  let outputQEls = [];
  let outputNameEls = [];
  let layerLabelEls = [];
  let focusEls = [];
  let current = null;
  let selected = { layer: 0, neuron: 0 };
  let lastEdgeKey = null;
  let resizeTimer = null;

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const key in attrs) {
      node.setAttribute(key, attrs[key]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function nodeY(layer, index) {
    if (layer === 0) {
      return layout.top + index * (layout.bottom - layout.top) /
        (SIZES[0] - 1);
    }
    if (layer === 1 || layer === 2) {
      return layout.top + index * (layout.bottom - layout.top) /
        (SIZES[layer] - 1);
    }
    return layout.outputY[index];
  }

  function pos(layer, index) {
    return [layout.X[layer], nodeY(layer, index)];
  }

  function attach(circle, layer, index) {
    circle.addEventListener('click', function () {
      select(layer, index);
    });
    circle.addEventListener('mousemove', function (event) {
      showTooltip(event, layer, index);
    });
    circle.addEventListener('mouseleave', hideTooltip);
  }

  function buildStatic() {
    const width = svg.parentElement
      ? svg.parentElement.clientWidth
      : window.innerWidth;

    layout = width < 1000 ? NARROW : WIDE;
    document.body.classList.toggle('compact', layout.narrow);
    svg.setAttribute('viewBox', layout.viewBox);

    svg.innerHTML = '';
    denseEdgeGroup = el('g', { 'aria-hidden': 'true' }, svg);
    denseEdgeKey = null;
    edgeGroup = el('g', {}, svg);
    nodeGroup = el('g', {}, svg);
    labelGroup = el('g', {}, svg);

    nodes = [[], [], [], []];
    inputValueEls = [];
    inputNameEls = [];
    outputQEls = [];
    outputNameEls = [];
    layerLabelEls = [];
    focusEls = [];

    for (let layer = 0; layer < 4; layer++) {
      const label = el('text', {
        x: layout.X[layer],
        y: layout.narrow ? 32 : 42,
        'text-anchor': 'middle',
        class: 'layerlabel',
        'font-size': layout.labelSize,
      }, labelGroup);
      label.textContent = layout.titles[layer];
      layerLabelEls.push(label);
    }

    // Input nodes (labels only in the wide layout).
    for (let i = 0; i < SIZES[0]; i++) {
      const y = nodeY(0, i);

      if (layout.inputLabels) {
        const value = el('text', {
          x: 8,
          y: y + 4,
          class: 'inputvalue',
        }, labelGroup);
        value.textContent = '0.00';
        inputValueEls.push(value);

        const name = el('text', {
          x: 208,
          y: y + 4,
          'text-anchor': 'end',
          class: 'inputname',
        }, labelGroup);
        name.textContent = FEATURE_NAMES[i];
        inputNameEls.push(name);
      }

      const circle = el('circle', {
        cx: layout.X[0],
        cy: y,
        r: layout.inputRadius,
        class: 'node',
        fill: '#2a303b',
        stroke: '#3a4150',
        'stroke-width': 0.8,
      }, nodeGroup);
      attach(circle, 0, i);
      nodes[0].push(circle);
    }

    // Hidden layers.
    for (let layer = 1; layer <= 2; layer++) {
      el('line', {
        x1: layout.X[layer],
        y1: layout.top - 8,
        x2: layout.X[layer],
        y2: layout.bottom + 8,
        stroke: '#232734',
        'stroke-width': 1,
      }, labelGroup);

      for (let i = 0; i < SIZES[layer]; i++) {
        const circle = el('circle', {
          cx: layout.X[layer],
          cy: nodeY(layer, i),
          r: layout.hiddenRadius,
          class: 'node',
          fill: '#2a303b',
          stroke: 'none',
        }, nodeGroup);
        attach(circle, layer, i);
        nodes[layer].push(circle);
      }
    }

    // Output nodes.
    for (let i = 0; i < SIZES[3]; i++) {
      const y = nodeY(3, i);

      const circle = el('circle', {
        cx: layout.X[3],
        cy: y,
        r: 14,
        class: 'node',
        fill: '#2a303b',
        stroke: '#3a4150',
        'stroke-width': 1,
      }, nodeGroup);
      attach(circle, 3, i);
      nodes[3].push(circle);

      const labelX = layout.outputLabelsRight
        ? layout.X[3] + 26
        : layout.X[3];
      const anchor = layout.outputLabelsRight ? 'start' : 'middle';

      const name = el('text', {
        x: labelX,
        y: layout.outputLabelsRight ? y - 4 : y + 30,
        'text-anchor': anchor,
        class: 'outputname',
      }, labelGroup);
      name.textContent = ACTION_NAMES[i];
      outputNameEls.push(name);

      const q = el('text', {
        x: labelX,
        y: layout.outputLabelsRight ? y + 14 : y + 47,
        'text-anchor': anchor,
        class: 'outputq',
      }, labelGroup);
      q.textContent = '0.000';
      outputQEls.push(q);
    }

    selectionRing = el('circle', {
      r: 6,
      fill: 'none',
      stroke: '#ffffff',
      'stroke-width': 2,
      opacity: 0,
      'pointer-events': 'none',
    }, nodeGroup);

    updateSelectionRing();
  }

  function selectNode(layer, neuron) {
    selected = { layer: layer, neuron: neuron };
    updateSelectionRing();

    if (window.Dashboard && window.Dashboard.fetchNow) {
      window.Dashboard.fetchNow();
    }
  }

  function select(nodeLayer, index) {
    if (nodeLayer === 0) return;
    selectNode(nodeLayer - 1, index);
  }

  function updateSelectionRing() {
    const nodeLayer = selected.layer + 1;
    const circle = nodes[nodeLayer] && nodes[nodeLayer][selected.neuron];
    if (!circle || !selectionRing) return;

    const radius = Number(circle.getAttribute('r'));
    selectionRing.setAttribute('cx', circle.getAttribute('cx'));
    selectionRing.setAttribute('cy', circle.getAttribute('cy'));
    selectionRing.setAttribute('r', radius + 3.5);
    selectionRing.setAttribute('opacity', 1);
  }

  function activationColor(value, maxValue) {
    if (!(value > 1e-9)) return '#2a303b';
    const t = Math.max(0, Math.min(1, value / (maxValue || 1)));
    return 'rgba(94, 230, 160, ' + (0.3 + 0.7 * t).toFixed(3) + ')';
  }

  function inputColor(value) {
    const t = Math.max(0, Math.min(1, Number(value)));
    if (!(t > 1e-9)) return '#2a303b';
    return 'rgba(108, 184, 255, ' + (0.25 + 0.75 * t).toFixed(3) + ')';
  }

  function outputColor(value, maxAbs) {
    if (!(Math.abs(value) > 1e-9)) return '#2a303b';
    const t = Math.min(1, Math.abs(value) / (maxAbs || 1));
    if (value > 0) {
      return 'rgba(255, 215, 94, ' + (0.3 + 0.7 * t).toFixed(3) + ')';
    }
    return 'rgba(255, 122, 122, ' + (0.3 + 0.7 * t).toFixed(3) + ')';
  }

  function render(data) {
    current = data;
    selected = {
      layer: data.selected.layer,
      neuron: data.selected.neuron,
    };

    const inputs = data.activations[0];
    for (let i = 0; i < inputs.length; i++) {
      nodes[0][i].setAttribute('fill', inputColor(inputs[i]));

      if (inputValueEls[i]) {
        inputValueEls[i].textContent = inputs[i].toFixed(2);
        inputValueEls[i].setAttribute(
          'fill',
          inputs[i] > 0 ? '#6cb8ff' : '#5a6273'
        );
      }
    }

    const hidden = [null, data.activations[1], data.activations[2]];
    for (let layer = 1; layer <= 2; layer++) {
      const values = hidden[layer];
      let maxValue = 0;
      for (let i = 0; i < values.length; i++) {
        if (values[i] > maxValue) maxValue = values[i];
      }
      for (let i = 0; i < values.length; i++) {
        nodes[layer][i].setAttribute(
          'fill',
          activationColor(values[i], maxValue)
        );
      }
    }

    const qValues = data.activations[3];
    let maxQ = 0;
    for (let i = 0; i < qValues.length; i++) {
      maxQ = Math.max(maxQ, Math.abs(qValues[i]));
    }
    for (let i = 0; i < qValues.length; i++) {
      nodes[3][i].setAttribute('fill', outputColor(qValues[i], maxQ));
      nodes[3][i].setAttribute(
        'stroke',
        i === data.chosen_action ? '#ffd75e' : '#3a4150'
      );
      nodes[3][i].setAttribute(
        'stroke-width',
        i === data.chosen_action ? 2.5 : 1
      );
      outputQEls[i].textContent = qValues[i].toFixed(3);
      outputNameEls[i].classList.toggle(
        'chosen',
        i === data.chosen_action
      );
    }

    updateSelectionRing();

    const dense = Boolean(data.all_weights);
    if (dense && denseEdgeKey !== String(data.version)) {
      drawDenseEdges(data.all_weights);
      denseEdgeKey = String(data.version);
    }
    denseEdgeGroup.style.display = dense ? '' : 'none';
    const key = String(data.version) + '|' + (layout.narrow ? 'n' : 'w') +
      '|' + selected.layer + '|' + selected.neuron + '|' + dense;
    if (key !== lastEdgeKey) {
      drawEdges(data);
      lastEdgeKey = key;
    }
  }

  function drawDenseEdges(weights) {
    const fragment = document.createDocumentFragment();
    weights.forEach(function (matrix, layer) {
      matrix.forEach(function (row, target) {
        const end = pos(layer + 1, target);
        row.forEach(function (weight, source) {
          const start = pos(layer, source);
          el('line', {
            x1: start[0],
            y1: start[1],
            x2: end[0],
            y2: end[1],
            stroke: weight >= 0 ? '#5ee6a0' : '#ff7a7a',
            'stroke-width': 0.4,
            'stroke-opacity': Math.min(0.18, 0.012 + 0.1 * Math.abs(weight)),
          }, fragment);
        });
      });
    });
    denseEdgeGroup.replaceChildren(fragment);
  }

  function drawEdges(data) {
    edgeGroup.innerHTML = '';
    focusEls = [];

    const maxAbs = [0, 0, 0];

    const backgroundEdges = data.all_weights ? [] : data.edges;
    backgroundEdges.forEach(function (edge) {
      maxAbs[edge.f] = Math.max(maxAbs[edge.f], Math.abs(edge.w));
    });
    data.focus_in.forEach(function (edge) {
      maxAbs[selected.layer] = Math.max(
        maxAbs[selected.layer],
        Math.abs(edge.w)
      );
    });
    data.focus_out.forEach(function (edge) {
      maxAbs[selected.layer + 1] = Math.max(
        maxAbs[selected.layer + 1],
        Math.abs(edge.w)
      );
    });

    backgroundEdges.forEach(function (edge) {
      const start = pos(edge.f, edge.s);
      const end = pos(edge.f + 1, edge.t);
      const magnitude = Math.abs(edge.w) / (maxAbs[edge.f] || 1);

      el('line', {
        x1: start[0],
        y1: start[1],
        x2: end[0],
        y2: end[1],
        stroke: edge.w >= 0 ? '#5ee6a0' : '#ff7a7a',
        'stroke-width': 0.7,
        'stroke-opacity': (0.035 + 0.20 * magnitude).toFixed(3),
        class: 'edge',
      }, edgeGroup);
    });

    data.focus_in.forEach(function (edge) {
      const start = pos(selected.layer, edge.s);
      const end = pos(selected.layer + 1, selected.neuron);
      const magnitude = Math.abs(edge.w) / (maxAbs[selected.layer] || 1);

      const line = el('line', {
        x1: start[0],
        y1: start[1],
        x2: end[0],
        y2: end[1],
        stroke: edge.w >= 0 ? '#5ee6a0' : '#ff7a7a',
        'stroke-width': (0.8 + 2.2 * magnitude).toFixed(2),
        'stroke-opacity': (0.35 + 0.55 * magnitude).toFixed(3),
        class: 'edge focus',
      }, edgeGroup);
      focusEls.push(line);
    });

    data.focus_out.forEach(function (edge) {
      const start = pos(selected.layer + 1, selected.neuron);
      const end = pos(selected.layer + 2, edge.t);
      const magnitude = Math.abs(edge.w) /
        (maxAbs[selected.layer + 1] || 1);

      const line = el('line', {
        x1: start[0],
        y1: start[1],
        x2: end[0],
        y2: end[1],
        stroke: edge.w >= 0 ? '#5ee6a0' : '#ff7a7a',
        'stroke-width': (0.6 + 1.8 * magnitude).toFixed(2),
        'stroke-opacity': (0.18 + 0.45 * magnitude).toFixed(3),
        class: 'edge focus',
      }, edgeGroup);
      focusEls.push(line);
    });
  }

  function showTooltip(event, layer, index) {
    if (!current) return;

    const activation = current.activations[layer][index];
    let html = '';

    if (layer === 0) {
      html = '<b>Input feature</b><br>' + FEATURE_NAMES[index] +
        '<br>value = ' + activation.toFixed(4);
    } else if (layer === 3) {
      html = '<b>' + ACTION_NAMES[index] + '</b><br>' +
        'Q-value = ' + activation.toFixed(4);
    } else {
      const z = current.zs[layer] ? current.zs[layer][index] : null;
      const bias = current.biases[layer]
        ? current.biases[layer][index]
        : null;

      html = '<b>' + (layer === 1 ? 'Hidden 1' : 'Hidden 2') +
        ' \u00b7 neuron ' + index + '</b><br>' +
        'z = ' + (z === null ? '-' : z.toFixed(4)) + '<br>' +
        'ReLU(z) = ' + activation.toFixed(4) + '<br>' +
        'bias = ' + (bias === null ? '-' : bias.toFixed(4));
    }

    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    moveTooltip(event);
  }

  function moveTooltip(event) {
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;

    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) {
      x = event.clientX - rect.width - pad;
    }
    if (y + rect.height > window.innerHeight - 8) {
      y = event.clientY - rect.height - pad;
    }

    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  function animate() {
    focusEls.forEach(function (line) {
      line.classList.add('flow');
      window.setTimeout(function () {
        line.classList.remove('flow');
      }, 2600);
    });

    layerLabelEls.forEach(function (label, index) {
      window.setTimeout(function () {
        label.classList.add('flash');
        window.setTimeout(function () {
          label.classList.remove('flash');
        }, 500);
      }, index * 420);
    });
  }

  function init() {
    svg = document.getElementById('net');
    tooltip = document.getElementById('tooltip');
    buildStatic();

    window.addEventListener('resize', function () {
      if (resizeTimer) window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(function () {
        const width = svg.parentElement
          ? svg.parentElement.clientWidth
          : window.innerWidth;
        const shouldBeNarrow = width < 1000;
        if (shouldBeNarrow !== layout.narrow) {
          lastEdgeKey = null;
          buildStatic();
          if (current) render(current);
        }
      }, 180);
    });
  }

  window.NetworkView = {
    init: init,
    render: render,
    animate: animate,
    select: select,
    selectNode: selectNode,
    getSelected: function () {
      return { layer: selected.layer, neuron: selected.neuron };
    },
    featureNames: FEATURE_NAMES,
  };
})();
