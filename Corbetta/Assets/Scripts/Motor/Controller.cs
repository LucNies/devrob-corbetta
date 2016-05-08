using UnityEngine;
using System.Collections;

public class Controller : MonoBehaviour {

    public Vector3 shoulderRotation = Vector3.zero;
    public float elbowRotation = 0;

    public GameObject upperArm, lowerArm;

    ShoulderJoint shoulder;
    Elbow elbow;
   

	// Use this for initialization
	void Start () {
        shoulder = upperArm.transform.GetComponent<ShoulderJoint>();
        
        elbow = lowerArm.transform.GetComponent<Elbow>();
        
	}
	
    void OnValidate()
    {
        try
        {
            elbow.SetAngle(elbowRotation);
            shoulder.ChangeRotation(shoulderRotation);
        }
        catch (System.Exception)
        {

        }
        
    }

}
