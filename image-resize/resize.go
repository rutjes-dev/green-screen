package image_resize

import (
	"github.com/nfnt/resize"
	"image"
	"math"
)

const InputSize = 513

func ResizeImage(image image.Image) image.Image {

	bounds := image.Bounds()
	resizeRatio := 1.0 * InputSize /
		math.Max(float64(bounds.Size().X), float64(bounds.Size().Y))

	tWidth := uint(resizeRatio * float64(bounds.Size().X))
	tHeight := uint(resizeRatio * float64(bounds.Size().Y))

	return resize.Resize(tWidth, tHeight, image, resize.Bilinear)
}
