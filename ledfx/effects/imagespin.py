import logging
import timeit
import urllib.request

import voluptuous as vol
from PIL import Image, ImageDraw

from ledfx.effects.twod import Twod
from ledfx.utils import get_icon_path

_LOGGER = logging.getLogger(__name__)


class Imagespin(Twod):
    NAME = "Image"
    CATEGORY = "Matrix"
    HIDDEN_KEYS = ["speed", "background_brightness", "mirror", "flip", "blur"]
    ADVANCED_KEYS = Twod.ADVANCED_KEYS + ["pattern"]

    start_time = timeit.default_timer()

    _power_funcs = {
        "Beat": "beat_power",
        "Bass": "bass_power",
        "Lows (beat+bass)": "lows_power",
        "Mids": "mids_power",
        "High": "high_power",
    }

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "pattern",
                description="use a test pattern",
                default=False,
            ): bool,
            vol.Optional(
                "frequency_range",
                description="Frequency range for the beat detection",
                default="Lows (beat+bass)",
            ): vol.In(list(_power_funcs.keys())),
            vol.Optional(
                "multiplier",
                description="Applied to the audio input to amplify effect",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "Min Size",
                description="The minimum size multiplier for the image",
                default=0.3,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "spin",
                description="spin image according to filter impulse",
                default=False,
            ): bool,
            vol.Optional(
                "clip",
                description="When spinning the image, force fit to frame, or allow clipping",
                default=False,
            ): bool,
            vol.Optional(
                "url source", description="Load image from", default=""
            ): str,
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        self.spin = 0.0

    def config_updated(self, config):
        super().config_updated(config)

        self.clip = self._config["clip"]
        self.min_size = self._config["Min Size"]
        self.power_func = self._power_funcs[self._config["frequency_range"]]
        self.init = True

    def audio_data_updated(self, data):
        # Get filtered bar power
        self.bar = (
            getattr(data, self.power_func)() * self._config["multiplier"] * 2
        )

    def do_once(self):
        if self._config["pattern"]:
            url_path = "https://images.squarespace-cdn.com/content/v1/60cc480d9290423b888eb94a/1624780092100-4FLILMIV0YHHU45GB7XZ/Test+Pattern+t.png"
        else:
            url_path = self._config["url source"]

        if url_path != "":
            try:
                with urllib.request.urlopen(url_path) as url:
                    self.bass_image = Image.open(url)
                    self.bass_image.thumbnail(
                        (self.t_width * 4, self.t_height * 4)
                    )
                _LOGGER.info(f"pre scaled {self.bass_image.size}")

                if self.bass_image.mode != "RGBA":
                    # If it doesn't have an alpha channel, create a new image with an alpha channel
                    image_with_alpha = Image.new(
                        "RGBA", self.bass_image.size, (255, 255, 255, 255)
                    )  # Create a white image with an alpha channel
                    image_with_alpha.paste(
                        self.bass_image, (0, 0)
                    )  # Paste the original image onto the new one
                    self.bass_image = image_with_alpha
            except Exception as e:
                _LOGGER.error(
                    f"Failed to load image from {self._config['url source']}: {e}"
                )
                self.bass_image = Image.open(get_icon_path("tray.png"))
        else:
            self.bass_image = Image.open(get_icon_path("tray.png"))
        self.init = False

    def draw(self):
        if self.init:
            self.do_once()

        rgb_image = Image.new("RGB", (self.t_width, self.t_height))
        rgb_draw = ImageDraw.Draw(rgb_image)

        if self.test:
            self.draw_test(rgb_draw)
            size = 1.0
            spin = 1.0
        else:
            size = self.bar + self.min_size
            spin = self.bar

        image_w = int(self.t_width * size)
        image_h = int(self.t_height * size)

        if image_w > 0 and image_h > 0:
            # make a copy of the original that we will manipulate
            bass_sized_img = self.bass_image.copy()

            if self._config["spin"]:
                self.spin += spin

                if self.spin > 360:
                    self.spin = 0.0
                bass_sized_img = bass_sized_img.rotate(
                    self.spin, expand=self.clip
                )

            # resize bass_image to fit in the target
            bass_sized_img.thumbnail(
                (image_w, image_h),
                Image.BILINEAR,
            )

            # render bass_sized_img into rgb_image centered with alpha
            rgb_image.paste(
                bass_sized_img,
                (
                    int((self.t_width - bass_sized_img.width) / 2),
                    int((self.t_height - bass_sized_img.height) / 2),
                ),
                bass_sized_img,
            )

        return rgb_image
