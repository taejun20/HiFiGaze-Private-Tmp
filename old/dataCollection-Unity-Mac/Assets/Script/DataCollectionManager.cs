using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using UnityEngine.UI;

public class DataCollectionManager : MonoBehaviour
{
    [SerializeField] Transform fixationTargetTransform;
    [SerializeField] TextMeshPro expLabel1;
    [SerializeField] RawImage backgroundImage;

    public bool showTargetLocations;

    float horizontalLocationCount = 1f;
    float verticalLocationCount = 2f;

    List<Vector2> targetLocationList = new List<Vector2>();
    int currentPositionIndex = 0;

    bool taskStarted = false;
    bool coroutineRunning = false;
    bool taskFinished = false;

    int currentSessionNumber = 1;

    WebCamTexture webcamTexture;
    int frameCounter = 0;
    int savedFrameCounter = 0;
    string imageSavePath;
    string csvFilePath;
    StreamWriter csvWriter;

    List<Color[]> cameraFrameList = new List<Color[]>();
    Texture2D snap;

    void Start()
    {
        targetLocationList.Clear();
        for (int i = 0; i < (int)horizontalLocationCount; i++)
        {
            for (int j = 0; j < (int)verticalLocationCount; j++)
            {
                Vector2 targetLocation = new Vector2(-0.96f + 1.92f / (horizontalLocationCount + 1f) * (i + 1), 0.56f - 1.08f / (verticalLocationCount + 1f) * (j + 1));
                targetLocationList.Add(targetLocation);
                if (showTargetLocations)
                {
                    GameObject go = Instantiate(fixationTargetTransform.gameObject, fixationTargetTransform.position, fixationTargetTransform.rotation);
                    go.transform.position = new Vector3(targetLocation.x, targetLocation.y, go.transform.position.z);
                }
            }
        }
        targetLocationList.Shuffle();
        targetLocationList.Shuffle();

        fixationTargetTransform.gameObject.SetActive(false);

        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webcamTexture = new WebCamTexture(devices[0].name, 1920, 1080);
            webcamTexture.Play();            
            StartCoroutine(WaitForWebcamInit());
        }
        else
        {
            Debug.LogError("No webcam found.");
            return;
        }

        //snap = new Texture2D(webcamTexture.width, webcamTexture.height);

        imageSavePath = Path.Combine("Assets/ExpData");
        csvFilePath = Path.Combine(imageSavePath, "frame_data.csv");

        if (!Directory.Exists(imageSavePath))
            Directory.CreateDirectory(imageSavePath);

        csvWriter = new StreamWriter(csvFilePath);
        csvWriter.WriteLine("FrameNumber,PositionX,PositionY,CurrentPositionIndex,ScreenImageNumber,TimeStamp");
    }

    IEnumerator WaitForWebcamInit()
    {
        // Wait until webcam is initialized and reports correct resolution
        while (webcamTexture.width <= 16 || webcamTexture.height <= 16)
            yield return null;

        Debug.Log($"Selected Camera: {webcamTexture.deviceName}");
        Debug.Log($"Actual resolution: {webcamTexture.width}x{webcamTexture.height}");

        snap = new Texture2D(webcamTexture.width, webcamTexture.height);
    }
    void Update()
    {
        if (!taskFinished)
        {
            if (!taskStarted)
            {
                if (Input.GetKeyDown(KeyCode.Space))
                {
                    fixationTargetTransform.gameObject.SetActive(true);
                    expLabel1.text = "";
                    StartCoroutine(MoveFromTo(fixationTargetTransform.gameObject, fixationTargetTransform.position, new Vector3(targetLocationList[0].x, targetLocationList[0].y, 0f)));
                    coroutineRunning = true;
                    taskStarted = true;
                }
            }
            else
            {
                if (!coroutineRunning)
                {
                    if (currentPositionIndex <= horizontalLocationCount * verticalLocationCount)
                    {
                        if (Input.GetKeyDown(KeyCode.Space))
                        {
                            expLabel1.text = "";
                            coroutineRunning = true;
                            StartCoroutine(WaitAndMove());
                        }
                    }
                }
            }


            if (coroutineRunning && webcamTexture.didUpdateThisFrame && currentPositionIndex != 0)
            {
                SaveCameraFrameIntoList();
            }
        }
    }

    void SaveCameraFrameIntoList()
    {
        Vector3 pos = fixationTargetTransform.position;
        string line = $"{frameCounter},{pos.x:F4},{pos.y:F4},{currentPositionIndex},{currentSessionNumber},{Time.time:F4}";
        cameraFrameList.Add(webcamTexture.GetPixels());
        csvWriter.WriteLine(line);
        csvWriter.Flush();
        frameCounter++;
    }

    public IEnumerator MoveFromTo(GameObject obj, Vector3 start, Vector3 end)
    {
        float speed = 0.2f;
        obj.transform.position = start;

        while (Vector3.Distance(obj.transform.position, end) > 0.01f)
        {
            obj.transform.position = Vector3.MoveTowards(obj.transform.position, end, speed * Time.deltaTime);
            yield return null;
        }

        obj.transform.position = end;
        coroutineRunning = false;

        if (currentPositionIndex >= horizontalLocationCount * verticalLocationCount)
        {
            if (currentSessionNumber < 4)
            {
                StartCoroutine(updateLabelBetweenSaveCameraFrameIntoFile("Wait for a minute, Take a rest for your eyes", "Press Space to Start Next"));
                currentSessionNumber++;
                Texture2D texture = Resources.Load<Texture2D>("background_laptop_" + currentSessionNumber.ToString());
                if (texture != null)
                    backgroundImage.texture = texture;

                currentPositionIndex = 0;
                targetLocationList.Shuffle();
                targetLocationList.Shuffle();
                taskStarted = false;
            }
            else
            {
                StartCoroutine(updateLabelBetweenSaveCameraFrameIntoFile("Finished, Wait until Saving", "Finished, Good to go"));
            }
        }
        else
        {
            currentPositionIndex++;
            expLabel1.text = "";
        }
    }

    IEnumerator WaitAndMove()
    {
        float waitTime = 0.0f;
        yield return new WaitForSeconds(waitTime);

        if (currentPositionIndex < (int)(horizontalLocationCount * verticalLocationCount))
        {
            StartCoroutine(MoveFromTo(fixationTargetTransform.gameObject,
                                      new Vector3(targetLocationList[currentPositionIndex - 1].x, targetLocationList[currentPositionIndex - 1].y, 0f),
                                      new Vector3(targetLocationList[currentPositionIndex].x, targetLocationList[currentPositionIndex].y, 0f)));
        }
        else
        {
            StartCoroutine(MoveFromTo(fixationTargetTransform.gameObject,
                                      new Vector3(targetLocationList[currentPositionIndex - 1].x, targetLocationList[currentPositionIndex - 1].y, 0f),
                                      new Vector3(targetLocationList[0].x, targetLocationList[0].y, 0f)));
        }
    }

    IEnumerator updateLabelBetweenSaveCameraFrameIntoFile(string beforeStr, string afterStr)
    {
        taskFinished = true;
        expLabel1.text = beforeStr;
        yield return null;
        SaveCameraFrameIntoFile();
        yield return null;
        expLabel1.text = afterStr;
        taskFinished = false;
    }

    void SaveCameraFrameIntoFile()
    {
        for (int i = 0; i < cameraFrameList.Count; i++)
        {
            snap.SetPixels(cameraFrameList[i]);
            snap.Apply();
            string imgPath = Path.Combine(imageSavePath, $"frame_{savedFrameCounter:D4}.png");
            try
            {
                File.WriteAllBytes(imgPath, snap.EncodeToPNG());
            }
            catch (IOException e)
            {
                Debug.LogError($"Failed to save image at {imgPath}: {e.Message}");
            }
            savedFrameCounter++;
        }
        cameraFrameList.Clear();
    }

    void OnApplicationQuit()
    {
        if (csvWriter != null)
        {
            csvWriter.Flush();
            csvWriter.Close();
        }

        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
    }
}
