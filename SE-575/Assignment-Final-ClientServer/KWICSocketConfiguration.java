public class KWICSocketConfiguration extends SocketConfiguration {

	public static String DEFAULT_HOST = "localhost";
	public static int DEFAULT_PORT = 12345;

	public KWICSocketConfiguration() {
		super(DEFAULT_HOST, DEFAULT_PORT);
	}

	public KWICSocketConfiguration(String host, int port) {
		super(host, port);
	}

}
