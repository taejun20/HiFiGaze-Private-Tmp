using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class HelperMethods
{
    public static void Shuffle<T>(this IList<T> list)
    {
        int n = list.Count;
        for (int i = 0; i < n - 1; i++)
        {
            int r = Random.Range(i, n); // UnityEngine.Random
            T temp = list[i];
            list[i] = list[r];
            list[r] = temp;
        }
    }
}
