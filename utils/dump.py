import multiprocessing as mp
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import speechset


class DumpReader(speechset.datasets.DataReader):
    """Dumped loader
    """
    def __init__(self, data_dir: str):
        """Initializer.
        Args:
            data_dir: path to the mother directory.
        """
        self.data_dir = data_dir
        self.speakers_, self.filelist, self.transcript = self.load_data(data_dir)

    def dataset(self) -> List[str]:
        """Return file reader.
        Returns:
            file-format datum read.er
        """
        return self.filelist
    
    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor.
        """
        return self.preprocessor

    def speakers(self) -> List[str]:
        """List of speakers.
        Returns:
            list of the speakers.
        """
        return self.speakers_

    def load_data(self, data_dir: str) -> Tuple[List[str], List[str], Dict[str, Tuple[int, str]]]:
        """Load the file lists.
        Args:
            data_dir: path to the mother directory.
        Returns:
            list of speakers, file paths and transcripts.
        """
        INTER = 'dumped'
        with open(os.path.join(data_dir, 'meta.json')) as f:
            meta = json.load(f)

        speakers = [info['name'] for info in meta.values()]
        filelists = [
            os.path.join(data_dir, INTER, filename)
            for filename in os.listdir(os.path.join(data_dir, INTER))
            if filename.endswith('.npy')]
        # transpose
        transcripts = {}
        for sid, info in meta.items():
            sid = int(sid)
            for (i, text, _) in info['lists']:
                transcripts[str(i)] = (sid, text)
        
        return speakers, filelists, transcripts

    def preprocessor(self, path: str) -> Tuple[int, str, np.ndarray]:
        """Load dumped.
        Args:
            path: str, path.
        Returns:
            tuple,
                sid: int, speaker id.
                text: str, text.
                audio: [np.float32; [T]], raw speech signal in range(-1, 1).
        """
        return tuple(np.load(path, allow_pickle=True))

    @staticmethod
    def dumper(args) -> Tuple[int, int, str, str]:
        """Dumper, multiprocessing purpose.
        Args:
            i: int, index of the datasets.
            path: str, path to the original datum.
            preproc: Callable, preprocessor.
            out_dir: path to the output directory.
            default_sid: default speaker id.
        Returns:
            i: index of the datasets.
            sid: speaker id.
            text: transcript.
            path: path to the original datum.
        """
        i, path, preproc, out_dir, default_sid = args
        outputs = preproc(path)
        assert len(outputs) in [2, 3]
        if len(outputs) == 2:
            text, audio = outputs
            outputs = default_sid, text, audio

        np.save(os.path.join(out_dir, f'{i}.npy'), outputs)

        sid, text, _ = outputs
        return i, sid, text, path

    @classmethod
    def dump(cls,
             reader: speechset.datasets.DataReader,
             out_dir: str,
             default_sid: int = -1,
             num_proc: Optional[int] = None,
             chunksize: int = 1):
        """Dump the reader.
        Args:
            reader: dataset reader.
            out_dir: path to the output directory.
            default_sid: default speaker id for unknown.
            num_proc: the number of the process for multiprocessing.
            chunksize: size of the imap_unordered chunk.
        """
        INTER = 'dumped'
        os.makedirs(os.path.join(out_dir, INTER), exist_ok=True)

        speakers = reader.speakers()
        dataset, preproc = reader.dataset(), reader.preproc()

        meta = {
            sid: {'name': speaker, 'lists': []}
            for sid, speaker in enumerate(speakers)}
        if num_proc is None:
            for i, path in enumerate(tqdm(dataset)):
                outputs = preproc(path)
                assert len(outputs) in [2, 3]
                if len(outputs) == 2:
                    text, audio = outputs
                    outputs = default_sid, text, audio
                    # lazy init
                    if default_sid not in meta:
                        meta[default_sid] = {'name': 'unknown', 'lists': []}

                np.save(os.path.join(out_dir, INTER, f'{i}.npy'), outputs)

                sid, text, _ = outputs
                meta[sid]['lists'].append((i, text, path))
        else:
            with mp.Pool(num_proc) as pool:
                worker = pool.imap_unordered(
                    DumpReader.dumper,
                    [
                        (i, path, preproc, os.path.join(out_dir, INTER), default_sid)
                        for i, path in enumerate(dataset)],
                    chunksize=chunksize)
                for i, sid, text, path in tqdm(worker, total=len(dataset)):
                    if sid == default_sid and default_sid not in meta:
                        meta[default_sid] = {'name': 'unknown', 'lists': []}
                    meta[sid]['lists'].append((i, text, path))

        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)


if __name__ == '__main__':
    def main():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--out-dir', required=True)
        parser.add_argument('--num-proc', default=None, type=int)
        parser.add_argument('--chunksize', default=1, type=int)
        parser.add_argument('--default-sid', default=-1, type=int)
        parser.add_argument('--sr', default=22050, type=int)
        args = parser.parse_args()

        # hard code the reader
        reader = speechset.datasets.ConcatReader([
            speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-100', args.sr),
            speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-360', args.sr),
            speechset.datasets.LibriSpeech('./datasets/LibriSpeech/train-other-500', args.sr),
            speechset.datasets.VCTK('./datasets/VCTK-Corpus', args.sr)])

        DumpReader.dump(
            reader,
            args.out_dir,
            args.default_sid,
            args.num_proc,
            args.chunksize)
        
    main()
