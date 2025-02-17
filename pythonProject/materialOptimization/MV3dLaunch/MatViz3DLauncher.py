import subprocess
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MatViz3DLauncher:
    def __init__(self, exe_path):
        """
        :param exe_path: Path to the MatViz3D executable file.
        """
        self.exe_path = exe_path

    def start(self, size, concentration, halfaxis_a, halfaxis_b, halfaxis_c,
              orientation_angle_a, orientation_angle_b, orientation_angle_c,
              wave_coefficient, output_file):
        """
        Run MatViz3D with the given parameters.

        :param size: Cube size.
        :param concentration: Concentration.
        :param halfaxis_a: Half-axis a.
        :param halfaxis_b: Half-axis b.
        :param halfaxis_c: Half-axis c.
        :param orientation_angle_a: Orientation angle a.
        :param orientation_angle_b: Orientation angle b.
        :param orientation_angle_c: Orientation angle c.
        :param wave_coefficient: Wave coefficient.
        :param output_file: Path to the output file.
        :return: Path to the output file or False in case of an error.
        """
        params = [
            self.exe_path,
            '--size', str(size),
            '--concentration', str(concentration),
            '--halfaxis_a', str(halfaxis_a),
            '--halfaxis_b', str(halfaxis_b),
            '--halfaxis_c', str(halfaxis_c),
            '--orientation_angle_a', str(orientation_angle_a),
            '--orientation_angle_b', str(orientation_angle_b),
            '--orientation_angle_c', str(orientation_angle_c),
            '--wave_coefficient', str(wave_coefficient),
            '--algorithm', 'Probability Algorithm',
            '--nogui',
            '--autostart',
            '--output', output_file
        ]

        try:
            result = subprocess.run(params, capture_output=True, text=True, check=True)
            logging.debug(f"Output result: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running the .exe: {e.stderr}")
            return False

        if not os.path.exists(output_file):
            logging.error(f"File {output_file} was not created.")
            return False

        return output_file
