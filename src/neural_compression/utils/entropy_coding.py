import torch
import numpy as np
from typing import List, Tuple, Dict
import pickle


class ArithmeticCoder:
    """산술 코딩 (간단한 구현)"""
    
    def __init__(self):
        self.precision = 32
        self.max_val = (1 << self.precision) - 1
    
    def encode(self, data: np.ndarray, probabilities: Dict[int, float]) -> bytes:
        """데이터 인코딩"""
        
        # 누적 확률 계산
        symbols = sorted(probabilities.keys())
        cumulative = {}
        cum_prob = 0.0
        
        for symbol in symbols:
            cumulative[symbol] = cum_prob
            cum_prob += probabilities[symbol]
        
        # 산술 코딩
        low = 0
        high = self.max_val
        
        encoded_data = []
        
        for symbol in data.flatten():
            symbol = int(symbol)
            if symbol in cumulative:
                range_size = high - low + 1
                
                # 새로운 범위 계산
                new_low = low + int(cumulative[symbol] * range_size)
                new_high = low + int((cumulative[symbol] + probabilities[symbol]) * range_size) - 1
                
                low = new_low
                high = new_high
        
        # 최종 코드
        code = (low + high) // 2
        
        # 메타데이터와 함께 반환
        result = {
            'code': code,
            'probabilities': probabilities,
            'shape': data.shape,
            'length': len(data.flatten())
        }
        
        return pickle.dumps(result)
    
    def decode(self, encoded: bytes) -> np.ndarray:
        """데이터 디코딩"""
        data = pickle.loads(encoded)
        
        # 여기서는 간단화를 위해 원본 모양만 반환
        # 실제로는 산술 디코딩 알고리즘 구현 필요
        return np.zeros(data['shape'])


class RangeCoder:
    """Range Coder (산술 코딩의 변형)"""
    
    def __init__(self):
        self.range_bits = 32
        self.range_max = (1 << self.range_bits) - 1
    
    def encode(self, data: torch.Tensor) -> bytes:
        """Range 인코딩"""
        
        # 히스토그램 계산
        data_flat = data.flatten().detach().cpu().numpy()
        unique_vals, counts = np.unique(data_flat, return_counts=True)
        
        # 확률 분포 추정
        probabilities = {}
        total_count = len(data_flat)
        
        for val, count in zip(unique_vals, counts):
            probabilities[float(val)] = count / total_count
        
        # 간단한 압축 (실제로는 range coding 구현)
        compressed = {
            'data': data_flat,
            'probabilities': probabilities,
            'shape': data.shape
        }
        
        return pickle.dumps(compressed)
    
    def decode(self, encoded: bytes) -> torch.Tensor:
        """Range 디코딩"""
        compressed = pickle.loads(encoded)
        
        data = torch.from_numpy(compressed['data']).view(compressed['shape'])
        return data


class HuffmanCoder:
    """허프만 코딩"""
    
    def __init__(self):
        self.code_table = {}
        self.decode_table = {}
    
    def build_tree(self, probabilities: Dict[int, float]):
        """허프만 트리 구축"""
        import heapq
        
        # 우선순위 큐 초기화
        heap = []
        for symbol, prob in probabilities.items():
            heapq.heappush(heap, (prob, symbol))
        
        # 트리 구축
        node_id = max(probabilities.keys()) + 1
        
        while len(heap) > 1:
            prob1, node1 = heapq.heappop(heap)
            prob2, node2 = heapq.heappop(heap)
            
            merged_prob = prob1 + prob2
            heapq.heappush(heap, (merged_prob, node_id))
            
            node_id += 1
        
        # 코드 테이블 생성 (간단화)
        self.code_table = {}
        for i, symbol in enumerate(probabilities.keys()):
            self.code_table[symbol] = bin(i)[2:].zfill(int(np.ceil(np.log2(len(probabilities)))))
        
        # 디코드 테이블
        self.decode_table = {v: k for k, v in self.code_table.items()}
    
    def encode(self, data: torch.Tensor) -> bytes:
        """허프만 인코딩"""
        data_np = data.detach().cpu().numpy().astype(int)
        
        # 확률 분포 계산
        unique_vals, counts = np.unique(data_np, return_counts=True)
        total_count = len(data_np.flatten())
        
        probabilities = {}
        for val, count in zip(unique_vals, counts):
            probabilities[int(val)] = count / total_count
        
        # 허프만 트리 구축
        self.build_tree(probabilities)
        
        # 인코딩
        encoded_bits = []
        for val in data_np.flatten():
            if int(val) in self.code_table:
                encoded_bits.append(self.code_table[int(val)])
        
        # 결과 패키징
        result = {
            'encoded_bits': ''.join(encoded_bits),
            'code_table': self.code_table,
            'shape': data.shape
        }
        
        return pickle.dumps(result)
    
    def decode(self, encoded: bytes) -> torch.Tensor:
        """허프만 디코딩"""
        data = pickle.loads(encoded)
        
        # 간단화를 위해 원본 모양의 영 텐서 반환
        return torch.zeros(data['shape'])