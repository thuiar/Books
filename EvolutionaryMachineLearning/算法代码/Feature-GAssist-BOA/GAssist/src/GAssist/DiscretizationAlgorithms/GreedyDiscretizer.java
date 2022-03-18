package GAssist.DiscretizationAlgorithms;

import java.util.Vector;

public class GreedyDiscretizer extends Discretizer {

	protected Vector discretizeAttribute(int attribute, int[] values,
			int begin, int end) {

		Vector cp = new Vector();

		int current = 0;
		while (current < values.length) {
			int next = current + 1;
			while (next < values.length
					&& realValues[attribute][values[current]] == realValues[attribute][values[next]]) {
				next++;
			}
			if (next < values.length) {
				double point = (realValues[attribute][values[current]] + realValues[attribute][values[next]]) / 2;
				cp.add(new Double(point));
			}
			current = next;
		}
		return cp;
	}
}
