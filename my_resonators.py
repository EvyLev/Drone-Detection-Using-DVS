from sctn.resonator import simple_resonator

clk_freq = 1536000

def resonator_105():
    resonator = simple_resonator(
        freq0=105,
        clk_freq=clk_freq,
        lf=4,
        thetas = [-14.609, -13.618, -11.787, -12.061],
        weights = [51.511, 22.744, 26.914, 23.81, 24.306]
    )
    return resonator

def resonator_110():
    resonator = simple_resonator(
        freq0=110,
        clk_freq=clk_freq,
        lf=4,
        thetas = [-12.327, -11.735, -9.997, -10.622],
        weights = [43.742, 19.585, 23.06, 20.334, 21.4]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_115():
    resonator = simple_resonator(
        freq0=115,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-13.016, -12.335, -10.737, -10.707],
        weights = [46.083, 20.491, 24.346, 21.682, 21.655],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_128():
    resonator = simple_resonator(
        freq0=128,
        clk_freq=clk_freq,
        lf=4,
        thetas = [-14.609, -13.618, -11.787, -12.061],
        weights= [51.511, 22.744, 26.914, 23.81, 24.306]
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_130():
    resonator = simple_resonator(
        freq0=130,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-38.705, -14.082, -11.844, -12.294],
        weights = [97.872, 21.307, 27.448, 24.055, 24.879]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_166():
    resonator = simple_resonator(
        freq0=166,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-20.0, -17.797, -15.956, -16.104],
        weights = [66.282, 27.3, 35.106, 32.001, 32.329],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_175():
    resonator = simple_resonator(
        freq0=175,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-24.442, -17.659, -17.053, -16.969],
        weights = [78.467, 29.833, 35.414, 34.069, 33.882]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_221():
    resonator = simple_resonator(
        freq0=221,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-26.511, -23.624, -21.416, -21.755],
        weights = [87.574, 36.152, 46.334, 42.9, 43.672],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_250():
    resonator = simple_resonator(
        freq0=250,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [ -30.365, -26.786, -24.465, -24.437],
        weights = [99.455, 40.538, 52.741, 48.947, 49.019]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_268():
    resonator = simple_resonator(
        freq0=268,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-8.927, -25.704, -25.206, -26.618],
        weights = [65.351, 47.423, 51.518, 50.371, 53.112],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_305():
    resonator = simple_resonator(
        freq0=305,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-37.124, -32.549, -29.814, -29.008],
        weights = [123.204, 50.851, 64.454, 59.25, 58.383],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_347():
    resonator = simple_resonator(
        freq0=347,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-44.809, -37.682, -34.249, -33.187],
        weights = [142.565, 55.948, 73.91, 68.354, 67.02],
        )
    return resonator
    
def resonator_402():
    resonator = simple_resonator(
        freq0=402,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-52.983, -43.606, -39.073, -37.241],
        weights = [170.532, 67.256, 86.596, 77.271, 75.25],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_436():
    resonator = simple_resonator(
        freq0=436,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-58.654, -47.541, -42.627, -40.103],
        weights = [187.055, 72.849, 94.515, 84.064, 81.135],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_462():
    resonator = simple_resonator(
        freq0=462,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-61.279, -49.554, -45.289, -44.119],
        weights = [194.982, 75.689, 97.964, 90.261, 89.009],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_477():
    resonator = simple_resonator(
        freq0=477,
        clk_freq=clk_freq,
        lf=4,
        thetas=[62.781, -51.17, -46.907, -45.359],
        weights=[200.524, 78.397, 101.374, 92.831, 91.37]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_509():
    resonator = simple_resonator(
        freq0=509,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-69.695, -55.164, -50.732, -47.095],
        weights = [219.153, 83.76, 109.522, 99.543, 95.232],
        )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_526():
    resonator = simple_resonator(
        freq0=526,
        clk_freq=clk_freq,
        lf=4,
        thetas=[-78.345, -57.279, -52.931, -48.217],
        weights=[239.144, 86.311, 114.419, 103.404, 97.135]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_545():
    resonator = simple_resonator(
        freq0=545,
        clk_freq=clk_freq,
        lf=4,
        thetas=[-81.139, -59.604, -54.623, -49.983],
        weights=[247.52, 89.124, 119.23, 106.752, 100.593]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_636():
    resonator = simple_resonator(
        freq0=636,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-98.054, -70.205, -64.087, -57.686],
        weights = [294.356, 103.068, 140.527, 124.915, 116.543]
    )
    resonator.log_out_spikes(-1)
    return resonator

def resonator_694():
    resonator = simple_resonator(
        freq0=694,
        clk_freq=clk_freq,
        lf = 4,
        thetas = [-113.694, -77.332, -69.207, -62.432],
        weights = [334.262, 112.405, 154.573, 135.118, 126.891]
    )
    resonator.log_out_spikes(-1)
    return resonator