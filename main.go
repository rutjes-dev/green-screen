package main

import (
	"bytes"
	"encoding/binary"
	"github.com/nfnt/resize"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	image_resize "green-screen/image-resize"
	"image"
	"image/color"
	"image/color/palette"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"os"
)

const (
	modelPath = "./deeplab/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"
	imagePath = "./img/binky.jpg"
)

func main() {

	// loading model
	model, err := ioutil.ReadFile(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	reader, err := os.Open(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()
	dog, err := jpeg.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	// reading the image
	doggie := image_resize.ResizeImage(dog)

	tensor, err := makeTensorFromImage(doggie)
	if err != nil {
		log.Fatal(err)
	}

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("ImageTensor").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("SemanticPredictions").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	shape := output[0].Shape()

	zeroImg := image.NewAlpha(image.Rect(0, 0, int(shape[2]), int(shape[1])))

	rgbImage := output[0].Value().([][][]int64)

	for y := 0; y < len(rgbImage[0])-1; y++ {
		for x := 0; x < len(rgbImage[0][y])-1; x++ {
			num := rgbImage[0][y][x]
			zeroImg.Set(x, y, palette.Plan9[num])
			if num == 0 {
				zeroImg.SetAlpha(x, y, color.Alpha{uint8(0)})
			} else {
				zeroImg.SetAlpha(x, y, color.Alpha{uint8(255)})
			}
		}
	}

	segmentedImage := resize.Resize(uint(dog.Bounds().Size().X), uint(dog.Bounds().Size().Y), zeroImg, resize.Bilinear)

	bounds := dog.Bounds()
	w, h := bounds.Max.X, bounds.Max.Y
	endresult := image.NewRGBA64(image.Rect(0, 0, w, h))
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			alphaColor := segmentedImage.At(x, y)
			imageColor := dog.At(x, y)
			rr, gg, bb, _ := imageColor.RGBA()
			_, _, _, alpha := alphaColor.RGBA()
			endresult.Set(x, y, color.NRGBA64{uint16(rr), uint16(gg), uint16(bb), uint16(alpha)})
		}
	}

	file, err := os.Create("result.png")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	png.Encode(file, endresult)
}

// Convert the image in filename to a Tensor suitable as input to the Inception model.
func makeTensorFromImage(m image.Image) (*tf.Tensor, error) {

	buf := new(bytes.Buffer)
	err := jpeg.Encode(buf, m, nil)
	if err != nil {
		return nil, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(buf.Bytes()))
	if err != nil {
		return nil, err
	}

	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToImage()
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func constructGraphToImage() (graph *tf.Graph, input, output tf.Output, err error) {

	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.Cast(s,
			op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Uint8),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func i64tob(val uint64) []byte {
	r := make([]byte, 8)
	for i := uint64(0); i < 8; i++ {
		r[i] = byte((val >> (i * 8)) & 0xff)
	}
	return r
}

func toUTF16(data []byte) ([]uint16, error) {
	bo := binary.BigEndian

	s := make([]uint16, 0, len(data)/2)
	for i := 2; i < len(data); i += 2 {
		s = append(s, bo.Uint16(data[i:i+2]))
	}
	return s, nil
}
