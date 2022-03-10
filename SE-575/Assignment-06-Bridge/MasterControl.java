public class MasterControl {

	public static void main(String[] args) {
		MasterControl controller = new MasterControl();
		controller.start(new PipeList(), new PipeQueue(), new PipeQueue());
	}

	public void start(Pipe p1, Pipe p2, Pipe p3) {

		Input input = new Input(null, p1);
		CircularShifter shifter = new CircularShifter(p1, p2);
		Alphabetizer sorter = new Alphabetizer(p2, p3);
		Output output = new Output(p3, null);

		input.start();
		shifter.start();
		sorter.start();
		output.start();

	}

}
