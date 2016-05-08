using UnityEngine;
using System.Collections;

public class Elbow : MonoBehaviour {

    public GameObject elbowJoint;

    float jointAngle = 0;
    float targetRotation = 0;


    public float GetAngle()
    {
        return jointAngle;
    }

    public void SetAngle(float angle)
    {
        targetRotation = angle;
    }
}
