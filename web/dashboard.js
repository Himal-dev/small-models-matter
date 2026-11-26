/* Dashboard polling, detail panels, charts and the demo board. */
(function () {
  'use strict';

  const ACTION_NAMES = ['STRAIGHT', 'TURN RIGHT', 'TURN LEFT'];
  const LAYER_ROW_LABELS = ['24 \u2192 128', '128 \u2192 128', '128 \u2192 3'];

  const scenarioSel = document.getElementById('scenario');
  const inspectLayer = document.getElementById('inspectLayer');
  const inspectNeuron = document.getElementById('inspectNeuron');
  const pauseBtn = document.getElementById('pause');
  const allConnections = document.getElementById('allConnections');
  const dot = document.getElementById('dot');
  const statusText = document.getElementById('statusText');
  const meta = document.getElementById('meta');

  let paused = false;
  let timer = null;
  let pollGeneration = 0;
  let requestId = 0;
  let weightsVersion = '';
  let allWeights = null;

  function num(value, digits) {
    if (digits === undefined) digits = 5;
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '-';
    }
    return Number(value).toFixed(digits);
  }

  function deltaHtml(value, digits) {
    if (digits === undefined) digits = 5;
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '<span class="d flat">no previous checkpoint</span>';
    }
    const cls = value > 0 ? 'up' : value < 0 ? 'down' : 'flat';
    const arrow = value > 0 ? '\u25b2 ' : value < 0 ? '\u25bc ' : '';
    const sign = value > 0 ? '+' : '';
    return '<span class="d ' + cls + '">' + arrow + sign +
      Number(value).toFixed(digits) + '</span>';
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>]/g, function (character) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
      }[character];
    });
  }

  function setStatus(kind, text) {
    dot.className = 'dot ' + kind;
    statusText.textContent = text;
  }

  async function fetchNow(advance = false) {
    const selected = window.NetworkView.getSelected();
    const scenario = scenarioSel.value;
    const id = ++requestId;
    const edgeQuery = allConnections.checked
      ? '&edge_version=' + encodeURIComponent(weightsVersion || 'initial')
      : '';

    try {
      if (window.SnakeReplay) {
        const data = await window.SnakeReplay.payload(selected, scenario, advance === true);
        if (id === requestId) render(data);
        return;
      }
      const response = await fetch(
        '/api/network?layer=' + selected.layer +
        '&neuron=' + selected.neuron +
        '&scenario=' + encodeURIComponent(scenario) + edgeQuery,
        { cache: 'no-store' }
      );
      const data = await response.json();

      if (id !== requestId) return;
      render(data);
    } catch (error) {
      if (id === requestId) {
        setStatus('error', window.SnakeReplay ? 'cannot load replay' : 'cannot reach the watcher');
      }
    }
  }

  function render(data) {
    if (!data.ok) {
      setStatus('error', data.error || 'no data');
      return;
    }

    if (data.all_weights) {
      allWeights = data.all_weights;
      weightsVersion = String(data.version);
    }
    data.all_weights = allConnections.checked && weightsVersion === String(data.version)
      ? allWeights : null;

    const age = data.checkpoint_age;
    if (data.version[2] === 'static') {
      setStatus('done', 'recorded replay · browser inference');
    } else if (data.version[2] === 'shipped') {
      setStatus('done', 'pretrained model · demo live');
    } else if (data.total_episodes && data.episode >= data.total_episodes) {
      setStatus('done', 'training complete \u00b7 demo live');
    } else if (age !== null && age < 150) {
      setStatus('live', 'live \u00b7 checkpoint ' + Math.round(age) + 's old');
    } else {
      setStatus('wait', 'waiting for the next episode');
    }

    if (data.error) {
      setStatus('wait', 'checkpoint read retry...');
    }

    const stats = data.stats || {};
    const episode = data.total_episodes
      ? data.episode + '/' + data.total_episodes
      : data.episode;

    meta.textContent =
      'episode ' + episode +
      '  \u00b7  score ' + (stats.score === null ? '-' : stats.score) +
      '  \u00b7  best ' + (stats.best_score === null ? '-' : stats.best_score) +
      '  \u00b7  eps ' + num(stats.epsilon, 3) +
      '  \u00b7  loss ' + num(stats.average_loss, 4);

    inspectLayer.value = String(data.selected.layer);
    const maxNeuron = data.selected.layer === 2 ? 2 : 127;
    inspectNeuron.max = maxNeuron;
    inspectNeuron.value = data.selected.neuron;

    window.NetworkView.render(data);
    renderSelectionSummary(data);
    renderInputList(data);
    renderNeuron(data);
    renderLayerStats(data);
    renderLearning(data);
    drawBoard(data.board);
    drawLayerChart(data.layer_history);
  }

  function renderSelectionSummary(data) {
    const container = document.getElementById('selectionSummary');
    if (!container) return;

    const selected = data.selected;
    const decision = selected.is_output
      ? (data.chosen_action === selected.neuron ? 'chosen' : 'not chosen')
      : (selected.activation > 0 ? 'active' : 'off');

    container.innerHTML =
      '<span class="ss-title">' +
        escapeHtml(selected.layer_label) + ' \u00b7 neuron ' +
        selected.neuron +
      '</span>' +
      '<span class="ss-item">bias <b>' + num(selected.bias, 4) +
        '</b></span>' +
      '<span class="ss-item">||w|| <b>' +
        num(selected.weight_norm, 3) + '</b></span>' +
      '<span class="ss-item">' +
        (selected.is_output ? 'Q' : 'z') + ' <b>' +
        num(selected.z, 3) + '</b></span>' +
      '<span class="ss-item">' +
        (selected.is_output ? 'decision' : 'ReLU') + ' <b>' +
        num(selected.is_output ? selected.z : selected.activation, 3) +
        '</b> (' + decision + ')</span>' +
      '<button id="jumpNeuron">equation &amp; weights \u2193</button>';

    const button = document.getElementById('jumpNeuron');
    if (button) {
      button.addEventListener('click', function () {
        const panel = document.getElementById('neuronPanel');
        if (panel) {
          panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    }
  }

  function renderInputList(data) {
    const container = document.getElementById('inputList');
    if (!container) return;

    const names = window.NetworkView.featureNames;
    const values = data.activations[0];

    container.innerHTML = names.map(function (name, index) {
      const value = values[index];
      const intensity = Math.min(1, Math.max(0, value));
      const color = value > 0
        ? 'rgba(108, 184, 255, ' +
          (0.3 + 0.7 * intensity).toFixed(3) + ')'
        : '#2a303b';

      return '<div class="inputItem">' +
        '<span class="inputDot" style="background:' + color + '"></span>' +
        '<span class="inputName">' + escapeHtml(name) + '</span>' +
        '<span class="inputVal">' + Number(value).toFixed(2) + '</span>' +
        '</div>';
    }).join('');
  }

  // -- learning story -----------------------------------------------------

  function renderLearning(data) {
    const learning = data.learning;
    if (!learning) return;

    const rewardTable = document.getElementById('rewardTable');
    if (rewardTable) {
      rewardTable.innerHTML = learning.rewards.map(function (row) {
        const cls = row.value > 0 ? 'pos' : row.value < 0 ? 'neg' : 'muted';
        const sign = row.value > 0 ? '+' : '';
        return '<div class="rewardRow">' +
          '<span class="rev">' + escapeHtml(row.event) + '</span>' +
          '<span class="rval ' + cls + '">' + sign +
            Number(row.value).toFixed(2) + '</span>' +
          '<span class="rwhy">' + escapeHtml(row.why) + '</span>' +
          '</div>';
      }).join('');
    }

    const hyper = document.getElementById('hyper');
    if (hyper) {
      const rows = [
        ['discount \u03b3', Number(learning.gamma).toFixed(2)],
        ['learning rate', String(learning.learning_rate)],
        ['batch size', String(learning.batch_size)],
        ['replay memory', Number(learning.memory_size).toLocaleString()],
        ['target refresh', 'every ' + learning.target_sync + ' updates'],
        ['grad clip', Number(learning.grad_clip).toFixed(1)],
        ['epsilon (now)', num(learning.epsilon, 3)],
        ['episode loss', num(learning.average_loss, 4)],
      ];

      hyper.innerHTML = rows.map(function (row) {
        return '<span class="k">' + row[0] + '</span>' +
          '<span class="v">' + row[1] + '</span>';
      }).join('');
    }

    const td = document.getElementById('tdLive');
    if (td && learning.demo) {
      const d = learning.demo;
      const tdClass = Math.abs(d.td_error) < 1e-9
        ? 'muted'
        : 'err';

      td.innerHTML =
        '<div class="tdRow"><span class="k">state s</span>' +
          '<span class="v">the demo board</span></div>' +
        '<div class="tdRow"><span class="k">action a</span>' +
          '<span class="v big">' + escapeHtml(d.action_name) + '</span></div>' +
        '<div class="tdRow"><span class="k">reward r</span>' +
          '<span class="v ' + (d.reward >= 0 ? 'pos' : 'neg') + '">' +
          (d.reward >= 0 ? '+' : '') + Number(d.reward).toFixed(2) +
          ' (' + escapeHtml(d.event) + ')</span></div>' +
        '<div class="tdRow"><span class="k">Q(s,a)</span>' +
          '<span class="v">' + Number(d.predicted).toFixed(3) + '</span></div>' +
        '<div class="tdRow"><span class="k">max Q(s\',a\')</span>' +
          '<span class="v">' + Number(d.best_future).toFixed(3) +
          (d.done ? ' (ignored)' : '') + '</span></div>' +
        '<div class="tdRow"><span class="k">target</span>' +
          '<span class="v big">' + Number(d.target).toFixed(3) +
          (d.done ? ' (terminal)' : '') + '</span></div>' +
        '<div class="tdRow"><span class="k">TD error</span>' +
          '<span class="v ' + tdClass + '">' +
          (d.td_error >= 0 ? '+' : '') +
          Number(d.td_error).toFixed(3) + '</span></div>' +
        '<div class="tdRow"><span class="k">squared error</span>' +
          '<span class="v">' + Number(d.squared_error).toFixed(4) +
          '</span></div>';
    } else if (td) {
      td.textContent = data.scenario === 'auto'
        ? 'Waiting for the next demo transition…'
        : 'Select the demo game to see a transition.';
    }

    const history = data.training_history || [];

    const lossPoints = history.filter(function (point) {
      return point.average_loss !== null &&
        point.average_loss !== undefined &&
        !Number.isNaN(Number(point.average_loss));
    }).map(function (point) {
      return { episode: point.episode, loss: point.average_loss };
    });

    const scorePoints = history.filter(function (point) {
      return point.score !== null && point.score !== undefined;
    }).map(function (point) {
      return {
        episode: point.episode,
        score: point.score,
        avg: point.average_score === null ? point.score : point.average_score,
      };
    });

    drawSeriesChart(
      document.getElementById('lossChart'),
      lossPoints,
      [{ key: 'loss', color: '#ff7a7a', label: 'MSE loss' }],
      {}
    );

    drawSeriesChart(
      document.getElementById('scoreChart'),
      scorePoints,
      [
        { key: 'score', color: '#5ee6a0', label: 'score' },
        { key: 'avg', color: '#6cb8ff', label: 'avg' },
      ],
      {}
    );
  }

  // -- selected neuron ----------------------------------------------------

  function equationHtml(selected) {
    const terms = selected.weights.map(function (weight, index) {
      const sign = weight >= 0 ? '+ ' : '- ';
      return sign + Math.abs(weight).toFixed(4) + '\u00b7' +
        selected.input_names[index];
    });

    let equation = 'z = ' +
      (selected.bias >= 0 ? '+ ' : '- ') +
      Math.abs(selected.bias).toFixed(4) + '\n';

    if (terms.length <= 32) {
      equation += '    ' + terms.join('\n    ');
    } else {
      equation += '    ' + terms.slice(0, 14).join('\n    ');
      equation += '\n    ... + ' + (terms.length - 14) + ' more terms';
    }

    equation += '\n\nz = ' + selected.z.toFixed(4);
    if (selected.is_output) {
      equation += '    Q = ' + selected.z.toFixed(4);
    } else {
      equation += '    ReLU(z) = ' + selected.activation.toFixed(4);
    }

    return escapeHtml(equation);
  }

  function renderNeuron(data) {
    const body = document.getElementById('neuronBody');
    const selected = data.selected;
    const output = selected.is_output;

    const zLabel = output ? 'Q-VALUE' : 'PRE-ACTIVATION z';
    const aLabel = output ? 'DECISION' : 'ACTIVATION ReLU(z)';

    const rows = selected.weights.map(function (weight, index) {
      const delta = selected.weight_deltas
        ? selected.weight_deltas[index]
        : null;
      const deltaClass = delta === null
        ? ''
        : delta > 0 ? 'pos' : delta < 0 ? 'neg' : '';
      const deltaText = delta === null
        ? '-'
        : (delta > 0 ? '+' : '') + delta.toFixed(5);

      return '<tr>' +
        '<td class="num">' + index + '</td>' +
        '<td>' + escapeHtml(selected.input_names[index]) + '</td>' +
        '<td class="num">' + weight.toFixed(5) + '</td>' +
        '<td class="num ' + deltaClass + '">' + deltaText + '</td>' +
        '<td class="num">' + selected.inputs[index].toFixed(4) + '</td>' +
        '<td class="num">' + selected.contributions[index].toFixed(5) +
        '</td>' +
        '</tr>';
    }).join('');

    const decisionText = output
      ? (data.chosen_action === selected.neuron
        ? 'largest Q \u00b7 action chosen'
        : 'not the largest Q')
      : (selected.activation > 0 ? 'active' : 'clamped to 0');

    body.innerHTML =
      '<div class="selhead">' +
        '<span class="badge">' + escapeHtml(selected.layer_label) + '</span>' +
        '<span class="selname">neuron ' + selected.neuron + '</span>' +
      '</div>' +
      '<div class="cards2">' +
        '<div class="mini">' +
          '<div class="k">BIAS b</div>' +
          '<div class="v">' + num(selected.bias) + '</div>' +
          '<div>' + deltaHtml(selected.bias_delta) + '</div>' +
        '</div>' +
        '<div class="mini">' +
          '<div class="k">WEIGHT NORM ||w||2</div>' +
          '<div class="v">' + num(selected.weight_norm, 4) + '</div>' +
          '<div>' + deltaHtml(selected.weight_norm_delta) + '</div>' +
        '</div>' +
        '<div class="mini">' +
          '<div class="k">' + zLabel + '</div>' +
          '<div class="v">' + num(selected.z, 4) + '</div>' +
          '<div class="muted">z = w . x + b</div>' +
        '</div>' +
        '<div class="mini">' +
          '<div class="k">' + aLabel + '</div>' +
          '<div class="v">' +
            num(output ? selected.z : selected.activation, 4) +
          '</div>' +
          '<div class="muted">' + decisionText + '</div>' +
        '</div>' +
      '</div>' +
      '<pre class="equation">' + equationHtml(selected) + '</pre>' +
      '<div class="tableWrap">' +
        '<table>' +
          '<thead><tr>' +
            '<th class="num">#</th>' +
            '<th>input</th>' +
            '<th class="num">weight</th>' +
            '<th class="num">delta</th>' +
            '<th class="num">x</th>' +
            '<th class="num">w . x</th>' +
          '</tr></thead>' +
          '<tbody>' + rows + '</tbody>' +
        '</table>' +
      '</div>' +
      '<div class="hint">Deltas are versus the previous checkpoint. ' +
        'Inputs are activations of the previous layer for the selected ' +
        'scenario.</div>' +
      '<canvas id="histChart" width="360" height="150"></canvas>' +
      '<div class="hint">Selected neuron: bias (yellow) and weight norm ' +
        '(green), each scaled to its own range.</div>';

    drawSeriesChart(
      document.getElementById('histChart'),
      data.history,
      [
        { key: 'bias', color: '#ffd75e', label: 'bias' },
        { key: 'weight_norm', color: '#5ee6a0', label: '||w||' },
      ],
      {}
    );
  }

  // -- whole-network stats ------------------------------------------------

  function renderLayerStats(data) {
    const container = document.getElementById('layerStats');
    const active = [data.active.h1, data.active.h2, null];

    container.innerHTML = data.layer_stats.map(function (stats, index) {
      const delta = stats.delta ? stats.delta.mean_abs : null;
      const activeText = index === 2
        ? 'chosen ' + ACTION_NAMES[data.active.output]
        : 'active ' + active[index] + '/128';

      return '<div class="layerrow">' +
        '<span class="lname">' + LAYER_ROW_LABELS[index] + '</span>' +
        '<span class="lval">mean |w| ' + stats.mean_abs.toFixed(3) +
          ' ' + deltaHtml(delta, 4) + '</span>' +
        '<span class="lval">' + activeText + '</span>' +
        '<span class="ldelta">max ' + stats.max_abs.toFixed(2) + '</span>' +
        '</div>';
    }).join('');
  }

  function drawLayerChart(history) {
    const points = (history || []).map(function (point) {
      return {
        episode: point.episode,
        l0: point.mean_abs[0],
        l1: point.mean_abs[1],
        l2: point.mean_abs[2],
      };
    });

    drawSeriesChart(
      document.getElementById('layerChart'),
      points,
      [
        { key: 'l0', color: '#ffd75e', label: '24\u2192128' },
        { key: 'l1', color: '#5ee6a0', label: '128\u2192128' },
        { key: 'l2', color: '#6cb8ff', label: '128\u21923' },
      ],
      { commonScale: true }
    );
  }

  // -- demo board ---------------------------------------------------------

  function drawBoard(board) {
    const canvas = document.getElementById('board');
    if (!canvas || !board) return;

    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cells = 20;
    const cell = size / cells;

    ctx.fillStyle = '#0e1015';
    ctx.fillRect(0, 0, size, size);

    ctx.strokeStyle = '#1b1f29';
    ctx.lineWidth = 1;
    for (let i = 1; i < cells; i++) {
      const offset = i * cell;
      ctx.beginPath();
      ctx.moveTo(offset, 0);
      ctx.lineTo(offset, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, offset);
      ctx.lineTo(size, offset);
      ctx.stroke();
    }

    if (board.food) {
      const fx = board.food[0] * cell + cell / 2;
      const fy = board.food[1] * cell + cell / 2;
      ctx.fillStyle = '#ff5c5c';
      ctx.beginPath();
      ctx.arc(fx, fy, cell * 0.28, 0, Math.PI * 2);
      ctx.fill();
    }

    board.snake.forEach(function (segment, index) {
      const x = segment[0] * cell;
      const y = segment[1] * cell;

      ctx.fillStyle = index === 0 ? '#6cffb0' : '#2f9f63';
      ctx.fillRect(x + 1.5, y + 1.5, cell - 3, cell - 3);

      if (index === 0) {
        ctx.strokeStyle = '#b8ffd9';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
      }
    });

    const action = board.action === null || board.action === undefined
      ? '-'
      : ACTION_NAMES[board.action];

    document.getElementById('boardMeta').textContent =
      'score ' + board.score + ' \u00b7 step ' + board.steps +
      (board.action === null || board.action === undefined
        ? ''
        : ' \u00b7 ' + action);

    document.getElementById('boardNote').textContent =
      scenarioSel.value === 'auto'
        ? (window.SnakeReplay ? 'Recorded board states with browser-computed actions.'
          : 'Live demo game played by the latest checkpoint, separate from training.')
        : 'Fixed sample state.';
  }

  // -- charts -------------------------------------------------------------

  function drawSeriesChart(canvas, points, series, options) {
    if (!canvas) return;

    options = options || {};
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 360;
    const H = canvas.clientHeight || 150;

    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const padL = 8;
    const padR = 8;
    const padT = 28;
    const padB = 18;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.font = '10px ui-monospace, monospace';
    let legendX = padL;

    series.forEach(function (item) {
      const latest = points && points.length
        ? points[points.length - 1][item.key]
        : null;

      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, 8, 10, 3);

      ctx.fillStyle = '#8b93a7';
      const label = item.label + ' ' +
        (latest === null || latest === undefined
          ? '-'
          : Number(latest).toFixed(3));
      ctx.fillText(label, legendX + 14, 13);

      legendX += ctx.measureText(label).width + 28;
    });

    if (!points || points.length === 0) {
      ctx.fillStyle = '#8b93a7';
      ctx.fillText('waiting for history...', padL, padT + plotH / 2);
      return;
    }

    const xs = points.map(function (point, index) {
      return point.episode || index + 1;
    });

    let minX = Math.min.apply(null, xs);
    let maxX = Math.max.apply(null, xs);
    const spanX = Math.max(1, maxX - minX);

    const allValues = [];
    const seriesValues = series.map(function (item) {
      const values = points.map(function (point) {
        return Number(point[item.key]);
      });
      values.forEach(function (value) {
        if (!Number.isNaN(value)) allValues.push(value);
      });
      return values;
    });

    let commonLo = Math.min.apply(null, allValues);
    let commonHi = Math.max.apply(null, allValues);
    if (commonHi - commonLo < 1e-9) {
      commonHi = commonLo + 1;
      commonLo -= 1;
    }

    function px(episode) {
      if (maxX === minX) return padL + plotW / 2;
      return padL + ((episode - minX) / spanX) * plotW;
    }

    function scale(values, common) {
      let lo;
      let hi;

      if (common) {
        lo = commonLo;
        hi = commonHi;
      } else {
        lo = Math.min.apply(null, values);
        hi = Math.max.apply(null, values);
        if (hi - lo < 1e-9) {
          hi = lo + 1;
          lo -= 1;
        }
      }

      const margin = (hi - lo) * 0.12;
      return [lo - margin, hi + margin];
    }

    function py(value, lo, hi) {
      return padT + (1 - (value - lo) / (hi - lo)) * plotH;
    }

    ctx.strokeStyle = '#232734';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = padT + (plotH * i) / 3;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    seriesValues.forEach(function (values, index) {
      const item = series[index];
      const bounds = scale(values, options.commonScale);
      const lo = bounds[0];
      const hi = bounds[1];

      ctx.strokeStyle = item.color;
      ctx.lineWidth = 1.7;
      ctx.beginPath();

      values.forEach(function (value, pointIndex) {
        const x = px(xs[pointIndex]);
        const y = py(value, lo, hi);
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      if (points.length <= 40) {
        ctx.fillStyle = item.color;
        values.forEach(function (value, pointIndex) {
          const x = px(xs[pointIndex]);
          const y = py(value, lo, hi);
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });

    ctx.fillStyle = '#8b93a7';
    ctx.textAlign = 'left';
    ctx.fillText('episode ' + minX, padL, H - 5);
    ctx.textAlign = 'right';
    ctx.fillText('episode ' + maxX, padL + plotW, H - 5);
    ctx.textAlign = 'left';
  }

  // -- polling ------------------------------------------------------------

  async function tick(generation) {
    await fetchNow(true);
    if (generation !== pollGeneration) return;
    timer = window.setTimeout(
      () => tick(generation),
      scenarioSel.value === 'auto' ? 700 : 1000
    );
  }

  pauseBtn.addEventListener('click', function () {
    paused = !paused;
    pollGeneration += 1;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';

    if (paused) {
      window.clearTimeout(timer);
      requestId += 1;
    } else {
      tick(pollGeneration);
    }
  });

  scenarioSel.addEventListener('change', function () {
    fetchNow();
  });

  allConnections.addEventListener('change', fetchNow);

  document.getElementById('animate').addEventListener('click', function () {
    window.NetworkView.animate();
  });

  function selectNeuron() {
    const layer = Number(inspectLayer.value);
    const max = layer === 2 ? 2 : 127;
    const neuron = Math.max(0, Math.min(max, Math.trunc(inspectNeuron.valueAsNumber || 0)));
    inspectNeuron.max = max;
    inspectNeuron.value = neuron;
    window.NetworkView.selectNode(layer, neuron);
  }

  inspectLayer.addEventListener('change', selectNeuron);
  inspectNeuron.addEventListener('change', selectNeuron);

  window.Dashboard = { fetchNow: fetchNow };

  window.NetworkView.init();
  tick(pollGeneration);
})();
