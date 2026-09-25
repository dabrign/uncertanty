"""
Synthetic test image generator: Creates a test scene with a cat on a sofa.
"""

import base64
import os
from io import BytesIO

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def generate_cat_on_sofa_image(output_path: str = "cat_on_sofa.jpg") -> bytes:
    """
    Generates a synthetic image of a orange cat resting on a blue sofa.
    Returns JPEG bytes and saves to output_path.
    """
    width, height = 600, 400

    if HAS_PIL:
        image = Image.new("RGB", (width, height), color=(240, 240, 245))
        draw = ImageDraw.Draw(image)

        # Draw Sofa Background (Blue)
        draw.rectangle([50, 150, 550, 350], fill=(41, 128, 185), outline=(21, 67, 96), width=3)
        draw.rectangle([70, 180, 530, 330], fill=(52, 152, 219))  # Cushions

        # Draw Orange Cat resting on cushion
        draw.ellipse([220, 210, 360, 300], fill=(230, 126, 34))  # Cat body
        draw.ellipse([330, 190, 390, 250], fill=(230, 126, 34))  # Cat head
        # Cat ears
        draw.polygon([(340, 195), (350, 170), (360, 195)], fill=(211, 84, 0))
        draw.polygon([(370, 195), (380, 170), (390, 195)], fill=(211, 84, 0))

        # Add text label
        try:
            font = ImageFont.load_default()
            draw.text(
                (20, 20),
                "Benchmark Scene: An orange cat resting on a blue sofa",
                fill=(44, 62, 80),
                font=font,
            )
            draw.text((250, 240), "CAT", fill=(255, 255, 255), font=font)
            draw.text((100, 290), "BLUE SOFA", fill=(255, 255, 255), font=font)
        except Exception:
            pass

        buf = BytesIO()
        image.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()
    else:
        # Fallback 1x1 valid minimal JPEG
        jpeg_bytes = base64.b64decode(
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP////////////////////////////////"
            "//////////////////////////////////////////////////////wgALCAABAAEBAREA"
            "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(jpeg_bytes)

    return jpeg_bytes


if __name__ == "__main__":
    path = "utils/cat_on_sofa.jpg"
    data = generate_cat_on_sofa_image(path)
    print(f"Generated synthetic test image: {path} ({len(data)} bytes)")
