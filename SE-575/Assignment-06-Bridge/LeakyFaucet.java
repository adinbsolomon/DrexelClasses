public class LeakyFaucet extends Pipe {

	public static String drop = "drip";

	@Override
	public void write(String string) {

	}

	@Override
	public String read() {
		return this.drop;
	}

	@Override
	public boolean isEmpty() {
		return false;
	}
}
