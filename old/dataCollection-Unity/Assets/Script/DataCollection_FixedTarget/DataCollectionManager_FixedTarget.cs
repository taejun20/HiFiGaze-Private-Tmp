using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using UnityEngine.UI;

public class DataCollectionManager_FixedTarget : MonoBehaviour
{
    [SerializeField]
    Transform fixationTargetTransform;

    [SerializeField]
    TextMeshPro expLabel1;

    [SerializeField]
    RawImage backgroundImage;

    public bool showTargetLocations;

    float horizontalLocationCount = 2f;
    float verticalLocationCount = 2f;

    List<Vector2> targetLocationList = new List<Vector2>();
    int currentPositionIndex = 0;

    int currentSessionNumber = 1;


    float fixationTargetLargestLocalScale = 0.025f;
    float fixationTargetSmallestLocalScale = 0.012f;


    // Webcam and logging
    WebCamTexture webcamTexture;
    int frameCounter = 0;
    int savedFrameCounter = 0;
    string imageSavePath;
    string csvFilePath;
    StreamWriter csvWriter;

    List<Color[]> cameraFrameList = new List<Color[]>();
    Texture2D snap;

    // Stopwatch for measuring trial duration
    System.Diagnostics.Stopwatch trialStopwatch = new System.Diagnostics.Stopwatch();
    bool firstTrialTimerDone = false;
    bool recordingStarted = false;
    bool userInitiatedTrial = false;
    bool taskTrialStarted = false;
    bool arrowShown = false;
    bool fileSaving = false;
    bool allFinished = false;
    int arrowDirection = -1;    // 0: Left, 1: Right

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

        // Webcam setup
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webcamTexture = new WebCamTexture(devices[0].name, 1920, 1080);
            webcamTexture.Play();
            snap = new Texture2D(webcamTexture.width, webcamTexture.height);
            Debug.Log($"Selected Camera: {webcamTexture.deviceName}");
            Debug.Log($"Actual resolution: {webcamTexture.width}x{webcamTexture.height}");
        }
        else
        {
            Debug.LogError("No webcam found.");
            return;
        }

        imageSavePath = Application.dataPath + "/ExpData/";
        csvFilePath = Application.dataPath + "/ExpData/frame_data.csv";

        if (!Directory.Exists(imageSavePath))
            Directory.CreateDirectory(imageSavePath);

        csvWriter = new StreamWriter(csvFilePath);
        csvWriter.WriteLine("FrameNumber,PositionX,PositionY,CurrentPositionIndex,ScreenImageNumber,TimeStamp");
    }


    void Update()
    {
        if (!fileSaving && !allFinished)
        {
            if (!taskTrialStarted)
            {
                if (arrowDirection == 0)
                {
                    if (Input.GetKeyDown(KeyCode.LeftArrow))
                        taskTrialStarted = true;
                }
                else if (arrowDirection == 1)
                {
                    if (Input.GetKeyDown(KeyCode.RightArrow))
                        taskTrialStarted = true;
                }
                else if (arrowDirection == -1)
                {
                    if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.RightArrow))
                        taskTrialStarted = true;
                }
            }

            if (taskTrialStarted)
            {
                Debug.Log("trialStopwatch.ElapsedMilliseconds / 1000f: " + (trialStopwatch.ElapsedMilliseconds / 1000f).ToString("f1"));
                if (!trialStopwatch.IsRunning)
                {
                    trialStopwatch.Reset();
                    trialStopwatch.Start();
                }

                if (currentPositionIndex == 0 && !firstTrialTimerDone)   // 3 seconds timer only for the first trial of each session
                {
                    float remainingTime = 2f - (trialStopwatch.ElapsedMilliseconds / 1000f);
                    expLabel1.text = remainingTime.ToString("f1") + " s";
                    if (remainingTime <= 0f)
                    {
                        firstTrialTimerDone = true;
                        expLabel1.text = "";
                        trialStopwatch.Reset();
                    }
                }
                else
                {
                    fixationTargetTransform.gameObject.SetActive(true);
                    fixationTargetTransform.position = new Vector3(targetLocationList[currentPositionIndex].x, targetLocationList[currentPositionIndex].y, 0f);

                    float elapsedSeconds = trialStopwatch.ElapsedMilliseconds / 1000f;
                    float halfPeriod = 0.6f;

                    float t = Mathf.PingPong(elapsedSeconds, halfPeriod) / halfPeriod;
                    float localScaleValue = Mathf.Lerp(fixationTargetSmallestLocalScale, fixationTargetLargestLocalScale, t);
                    fixationTargetTransform.GetChild(1).localScale = new Vector3(localScaleValue, localScaleValue, 0.001f);

                    if (!recordingStarted)
                    {
                        if (elapsedSeconds > 1.0f && elapsedSeconds < 2.0f)
                        {
                            recordingStarted = true;
                        }
                    }
                    else
                    {
                        SaveCameraFrameIntoList();
                    }

                    if (!arrowShown)
                    {
                        if (elapsedSeconds >= 2.0f)
                        {
                            // show left or right arrow upon the fixation target for 0.05s
                            arrowDirection = Random.Range(0, 2);    // randomly returns 0 or 1. 0: Left, 1: Right Arrow
                            if (arrowDirection == 0)
                                fixationTargetTransform.GetChild(2).gameObject.SetActive(true);
                            else if (arrowDirection == 1)
                                fixationTargetTransform.GetChild(3).gameObject.SetActive(true);

                            recordingStarted = false;
                            arrowShown = true;
                        }
                    }
                    else
                    {
                        if (elapsedSeconds >= 2.05f)
                        {
                            fixationTargetTransform.GetChild(2).gameObject.SetActive(false);
                            fixationTargetTransform.GetChild(3).gameObject.SetActive(false);
                            fixationTargetTransform.gameObject.SetActive(false);
                            trialStopwatch.Stop();
                            trialStopwatch.Reset();
                            arrowShown = false;
                            taskTrialStarted = false;
                            currentPositionIndex++;

                            if (currentPositionIndex == (int)(horizontalLocationCount * verticalLocationCount))
                            {
                                if (currentSessionNumber == 4)
                                {
                                    StartCoroutine(updateLabelBetweenSaveCameraFrameIntoFile("Finished, Wait until Saving", "Finished, Good to go"));
                                    allFinished = true;
                                }
                                else
                                {
                                    firstTrialTimerDone = false;
                                    currentPositionIndex = 0;
                                    currentSessionNumber++;
                                    targetLocationList.Shuffle();
                                    targetLocationList.Shuffle();
                                    Texture2D texture = Resources.Load<Texture2D>("background_laptop_" + currentSessionNumber.ToString());
                                    if (texture != null)
                                        backgroundImage.texture = texture;
                                    arrowDirection = -1;

                                    StartCoroutine(updateLabelBetweenSaveCameraFrameIntoFile("Wait for a minute, Take a rest for your eyes", "Press Arrows to Start Next"));
                                }
                            }
                        }
                    }
                }
            }
        }       
    }

    void SaveCameraFrameIntoList()
    {
        Vector3 pos = fixationTargetTransform.position;
        string line = $"{frameCounter},{pos.x:F4},{pos.y:F4},{currentPositionIndex},{currentSessionNumber},{Time.time:F4}";

        //Texture2D snap = new Texture2D(webcamTexture.width, webcamTexture.height);
        //snap.SetPixels(webcamTexture.GetPixels());
        //snap.Apply();

        //string imgPath = imageSavePath + "frame_" + frameCounter.ToString("D4") + ".png";
        //File.WriteAllBytes(imgPath, snap.EncodeToPNG());
        cameraFrameList.Add(webcamTexture.GetPixels());


        csvWriter.WriteLine(line);
        csvWriter.Flush();

        //Destroy(snap);
        frameCounter++;
    }

    IEnumerator updateLabelBetweenSaveCameraFrameIntoFile(string beforeStr, string afterStr)
    {
        fileSaving = true;
        expLabel1.text = beforeStr;

        // Wait one frame so UI can update
        yield return null;

        // Now do the heavy operation
        SaveCameraFrameIntoFile();

        // Wait one frame so UI can update
        yield return null;

        expLabel1.text = afterStr;
        fileSaving = false;
    }

    void SaveCameraFrameIntoFile()
    {
        for (int i = 0; i < cameraFrameList.Count; i++)
        {
            snap.SetPixels(cameraFrameList[i]);
            snap.Apply();

            string imgPath = imageSavePath + "frame_" + savedFrameCounter.ToString("D4") + ".png";
            File.WriteAllBytes(imgPath, snap.EncodeToPNG());

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
