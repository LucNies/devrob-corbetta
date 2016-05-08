using UnityEngine;
using System.Collections;

public class ShoulderJoint : MonoBehaviour {

    public float jointAngle = 0;
    public GameObject torso;
    public GameObject lowerArm;
    public GameObject elbow;
    public GameObject hand;
    public GameObject wrist;

	
    public void ChangeRotation(Vector3 rotation)
    {
        transform.eulerAngles = rotation;

        lowerArm.transform.position = elbow.transform.position;
        lowerArm.transform.eulerAngles = elbow.transform.eulerAngles;

        lowerArm.transform.RotateAround(elbow.transform.position, elbow.transform.up, 
        lowerArm.GetComponent<Elbow>().GetAngle());

        hand.transform.position = wrist.transform.position;
        hand.transform.eulerAngles = wrist.transform.eulerAngles;

    }
}
