import numpy as np
import voluptuous as vol
import time

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect

#First attempt at reimplementing https://gist.github.com/kriegsman/756ea6dcae8e30845b5a / twinklefox_base from https://github.com/Aircoookie/WLED/blob/main/wled00/FX.cpp
class Twinklefox(AudioReactiveEffect, HSVEffect):
    NAME = "Twinklefox"
    CATEGORY = "Atmospheric"

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "speed",
                description="Effect Speed",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "phase_peak",
                description="Phase peak",
                default=0.33,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.00001, max=1.0)),
            vol.Optional(
                "density",
                description="Twinkle density",
                default=0.625,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.00001, max=1.0)),
            vol.Optional(
                "reactivity",
                description="Audio Reactive modifier",
                default=0.2,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.00001, max=1.0)),
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)

        #Original twinklefox has a clock that runs in integer ticks, but goes to 2**32 milliseconds from start, or 1193 hours = 49 days
        #I guess we MIGHT have such a huge uptime so we should probably wrap our clock at some point, let's do two days
        self.clkwrap = 2*24*3600*1000.0

    def on_activate(self, pixel_count):
        self.last_time = time.time_ns()

        #clock, start with a random offset of up to 65536 milliseconds
        self.clk = np.random.rand(self.pixel_count)*65536.0
        #Original twinklefox generates these using a PRNG to reduce memory in such a way that it is deterministic for each call of the updater
        #in order to save memory.  We're not running on a microcontroller, so pregenerate our random per-pixel modifiers from 1.0 to 3.0
        self.speed_modifier = 1.0 + 2.0*np.random.rand(self.pixel_count)

        #cycle salt, also our hue modifier
        self.cycle_salt = np.random.randint(256, size=self.pixel_count)


    def config_updated(self, config):
        self._lows_power = 0
        self._lows_filter = self.create_filter(
            alpha_decay=0.05, alpha_rise=0.2
        )

        #Speed modifier is a milliseconds divisor between 3 and 72 in the original twinklefox implementation
        #with the default of 128 mapping to a divisor of 16, and the mapping being nonlinear
        #Instead let's use np.power(2,1+3*speed) for a min of 2 and a max of 128
        self.clk_div = np.power(2, 7 - 6*self._config["speed"])
        self.twinkle_dens = self._config["density"] #can we just reference this directly in render_hsv?

    def audio_data_updated(self, data):
        self._lows_power = self._lows_filter.update(
            data.lows_power(filtered=False)
        )

    def array_sawtooth(self, a):
        pk = self.config["phase_peak"] #I don't want to type this over and over again...
        return np.where(a < pk, a/pk, 1.0-(a-pk)/(1-pk))

    def render_hsv(self):
        now_ns = time.time_ns()
        dt = (now_ns - self.last_time)/1.0e6 #Original twinklefox ticks in milliseconds
        self.clk += (self.speed_modifier+self._config["reactivity"]*self._lows_power*250)*dt
        #Wrap our clock every 2 days just in case
        self.clk = np.where(self.clk > self.clkwrap, self.clk - self.clkwrap, self.clk)
        self.last_time = now_ns

        #Having partial ticks is probably overkill, but may as well do it
        #Original twinklefox uses integer ticks
        ticks = self.clk / self.clk_div

        #256 ticks per cycle
        (phase, cycles) = np.modf(ticks/256.0)

        #slowcycle16 in original twinklefox, randomized further using their PRNG algo and some sin() magic
        cycles16 = (cycles + self.cycle_salt).astype(np.uint16)
        #this emulates FastLED's sin8() wave function - not sure why it's there, I was hoping it would randomize hues better but that doesn't seem to be the case...
        sinoffset = (cycles16 % 0xff)/256.0
        self.array_sin(sinoffset)

        cycles16 += (sinoffset*256.0).astype(np.uint16)

        #The PRNG algo
        cycles16 = (cycles16 * 2053) + 1384

        #slowcycle8 in original implementation
        cycles8 = (cycles16 & 0xff).astype(np.uint8) + (cycles16 >> 8).astype(np.uint8)

        cycle_mod = cycles8/256.0

        #Handle the twinkle density factor
        bright = np.where(cycle_mod < self.twinkle_dens, self.array_sawtooth(phase), 0)

        hue_mod = self.cycle_salt/256.0
        #hue modifier with wraparound
        hue = np.where(cycle_mod > hue_mod, cycle_mod - hue_mod, cycle_mod + 1.0 - hue_mod)

        self.hsv_array[:, 0] = hue
        self.hsv_array[:, 1] = 1
        self.hsv_array[:, 2] = bright