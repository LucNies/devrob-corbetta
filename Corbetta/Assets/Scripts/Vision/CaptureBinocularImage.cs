using UnityEngine;
using System.Collections;
using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;

public class CaptureBinocularImage : MonoBehaviour
{

    public Camera leftEye;
    public Camera rightEye;
    private int counter = 0;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (counter % 60 == 0)
        {
            byte[] leftField = GetVisualField(leftEye);
            byte[] rightField = GetVisualField(rightEye);
            Mat img = new Mat(200, 400, DepthType.Cv8U, 3);

            img.SetTo(new Bgr(255, 0, 0).MCvScalar);

            CvInvoke.PutText(
            img,
            "Hello, world",
            new System.Drawing.Point(10, 80),
            FontFace.HersheyComplex,
            1.0,
            new Bgr(0, 255, 0).MCvScalar);

            // ImageViewer.Show(img, "Test Window");


        }
    }

    private byte[] GetVisualField(Camera camera)
    {
        int width = camera.pixelWidth;
        int height = camera.pixelHeight;
        Debug.Log("Taking image");
        RenderTexture rt = new RenderTexture(width, height, 24);
        camera.targetTexture = rt;
        Texture2D screen = new Texture2D(width, height, TextureFormat.RGB24, false);
        camera.Render();
        RenderTexture.active = rt;
        screen.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        camera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        byte[] bytes = screen.EncodeToPNG();
        string filename = ScreenShotName(width, height);
        System.IO.File.WriteAllBytes(filename, bytes);
        Debug.Log(string.Format("took screenshot {0}", filename));

        return bytes;
    }

    public static string ScreenShotName(int width, int height)
    {
        return string.Format("{0}/screenshots/screen_{1}x{2}_{3}.png",
                             Application.dataPath,
                             width, height,
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}