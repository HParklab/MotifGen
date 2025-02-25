#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from pepscore.model.models import MyModel1, MyModel1_motif  

# 사용할 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, args):
    model = MyModel1_motif(
        rec_features=21,
        pep_features=27,
        edge_features=67,
        motif_features=6,
        emb_features=args.emb_features,
        esm_emb_features=args.esm_emb_features,
        out_features=args.out_features,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        alpha=args.alpha,
        num_layers=args.num_layers
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

import os
def run_inference(model, input_data_path=None):
    if input_data_path:

        rec_node_feat = os.path.join(input_data_path, "rec_node_feat.npy")
        pep_node_feat = os.path.join(input_data_path, "pep_node_feat.npy")
        edge_feat = os.path.join(input_data_path, "edge_feat.npy")
        motif_score = os.path.join(input_data_path, "motif.npy")

        rec_node_feat = torch.tensor(np.load(rec_node_feat)).float().to(device)
        pep_node_feat = torch.tensor(np.load(pep_node_feat)).float().to(device)
        edge_feat = torch.tensor(np.load(edge_feat)).float().to(device)
        motif_score = torch.tensor(np.load(motif_score)).float().to(device)
        print("rec_node_feat.shape", rec_node_feat.shape)
        print("pep_node_feat.shape", pep_node_feat.shape)
        print("edge_feat.shape", edge_feat.shape)
        print("motif_score.shape", motif_score.shape)

        #all unsqueeze to add batch dimension
        rec_node_feat = rec_node_feat.unsqueeze(0)
        pep_node_feat = pep_node_feat.unsqueeze(0)
        edge_feat = edge_feat.unsqueeze(0)
        motif_score = motif_score.unsqueeze(0)
        """
        data = np.load(input_data_path)
        #
        pep_node_feat = torch.tensor(data['pep_node_feat']).float().to(device)
        rec_node_feat = torch.tensor(data['rec_node_feat']).float().to(device)
        edge_feat    = torch.tensor(data['edge_feat']).float().to(device)
        motif_score  = torch.tensor(data['motif_score']).float().to(device)
        """
    else:
        # 더미 데이터 (배치 사이즈 1)
        pep_node_feat = torch.randn(1, 27).to(device)
        rec_node_feat = torch.randn(1, 21).to(device)
        edge_feat    = torch.randn(1, 67).to(device)
        motif_score  = torch.randn(1, 6).to(device)
        
    with torch.no_grad():
        # 모델의 forward 결과: feature와 로짓을 반환한다고 가정
        feature, logits = model(pep_node_feat, rec_node_feat, edge_feat, motif_score)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    return probs.cpu().numpy(), preds.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="저장된 모델 체크포인트 경로")
    parser.add_argument("--input", type=str, default=None, help="입력 데이터 npz 파일 경로 (없으면 더미 데이터 사용)")
    # 모델 파라미터 (training 시 사용한 값과 동일해야 함)
    parser.add_argument("--emb_features", type=int, default=20)
    parser.add_argument("--esm_emb_features", type=int, default=16)
    parser.add_argument("--out_features", type=int, default=16)
    parser.add_argument("--attention_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--num_layers", type=int, default=1)
    
    args = parser.parse_args()
    
    # 모델 로드
    model = load_model(args.checkpoint, args.motif, args)
    # 추론 실행
    probs, preds = run_inference(model, args.input)
    
    print("예측 확률:", probs)
    print("예측 클래스 (0 또는 1):", preds)