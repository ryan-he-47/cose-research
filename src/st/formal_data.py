"""2026-10-03：已审批正式分布，磁盘索引与可复现的场景配方。"""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np

from src.formal_scene_tasks import CAPTION_VERSION, SCENE_VERSION, build_formal_scene, instruction, make_recipe
from src.page_document import PageLayoutConfig, build_document, layout_text_pages, sketch_page
from src.text_augmentation import TextAugmentationConfig, augment_rendered_text
from src.text_renderer import VectorFont
from .formal_assets import IndexedRawSource, IndexedStories, OffsetJSONReader
from .pipeline_pretrain import PretrainDocumentDataset


class RecipeSource:
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.reader = OffsetJSONReader()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        return self.reader.read(self.root / 'recipes.jsonl', self.offsets[index])

    def __getstate__(self):
        return {"root": str(self.root)}

    def __setstate__(self, state):
        self.__init__(**state)


class FormalDocumentDataset(PretrainDocumentDataset):
    def __init__(self, font, stories, drawings, *, font_path, scene_index=None,
                 scene_recipes=1000000, seed=42, split="train", augmentation=None,
                 validation_fraction=.02,object_sampling='with_replacement', paired_scene_share=0.):
        # 构造器只借用既有任务/增强协议；磁盘故事源随后替换，避免转成大tuple。
        super().__init__(font, (), drawings, seed=seed, split=split,
                         augmentation=augmentation, validation_fraction=validation_fraction)
        self.stories = stories
        self.tasks = ("T1", "O1", "O2", "S1", "S2", "S3", "S4", "S5")
        self.font_path = str(font_path)
        if object_sampling not in ('with_replacement','affine_pass_v1'):raise ValueError('未知对象采样方案')
        self.object_sampling=object_sampling
        if not 0 <= paired_scene_share <= 1: raise ValueError('配对份额须在0到1之间')
        self.paired_scene_share = float(paired_scene_share)
        self.recipes = RecipeSource(scene_index) if scene_index else None
        self.recipe_count = len(self.recipes) if self.recipes else scene_recipes
        if self.recipes and self.recipes.info["source"] != drawings.manifest():
            raise ValueError("场景配方与当前RAW索引来源不同")

    def get_for_task(self, task, index):
        if task not in self.tasks or index < 0:
            raise ValueError(f"无效任务/索引：{task}/{index}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.tasks.index(task), index]))
        augment = partial(augment_rendered_text, rng=rng, config=self.augmentation)
        augment = augment if self.augmentation.enabled else None
        text = lambda value, target=False: layout_text_pages(value, self.font,
            targetable=target, augment=augment)
        metadata = {"task": task, "task_index": index, "formal_version": SCENE_VERSION}
        if task == "T1":
            # 固定轮次的仿射置换，避免每轮都从同一篇故事开头读取。
            size = len(self.stories)
            multiplier = 104729
            import math
            while math.gcd(multiplier, size) != 1:
                multiplier += 2
            item = self.stories[(index*multiplier + self.seed) % size]
            metadata.update(source_id=item.source_id, text=item.text)
            return build_document(text(item.text, True), complete=True, metadata=metadata)
        if task.startswith("S"):
            # 五种任务共用相同的有限配方；文字和问答随机数独立于场景几何。
            recipe_id = index % self.recipe_count
            recipe = self.recipes[recipe_id] if self.recipes else make_recipe(self.drawings, self.seed, recipe_id)
            if task in ('S3','S4','S5') and self.paired_scene_share:
                # 每两个连续索引共用配方/问题；任务消费索引本身就是完整恢复游标。
                pair_id, variant = divmod(index, 2)
                select = np.random.default_rng(np.random.SeedSequence([self.seed, 2701, pair_id]))
                if select.random() < self.paired_scene_share:
                    from src.visual_grounding_scenes import paired_scene
                    pair_recipe_id = pair_id % self.recipe_count
                    paired = self.recipes[pair_recipe_id] if self.recipes else make_recipe(self.drawings, self.seed, pair_recipe_id)
                    if len({o['label'] for o in paired['objects']}) > 1:
                        document = paired_scene(task, paired, self.drawings, self.font,
                                                pair_id=pair_id, variant=variant, augment=augment)
                        document.metadata.update(metadata)
                        return document
            document = build_formal_scene(task, recipe, self.drawings, self.font, rng=rng, augment=augment)
            document.metadata.update(metadata)
            return document
        if self.object_sampling=='affine_pass_v1':
            size=len(self.drawings);multiplier=104729
            import math
            while math.gcd(multiplier,size)!=1:multiplier+=2
            drawing=self.drawings.get((index*multiplier+self.seed)%size)
        else:
            cls = index % len(self.drawings.classes)
            entries = self.drawings.class_indices[cls]
            drawing = self.drawings.get(int(entries[int(rng.integers(len(entries)))]))
        label = drawing.label
        metadata.update(source_id=drawing.key_id, label=label)
        if task == "O1":
            prompt = instruction(f"Draw: {label}", (f"Draw a sketch of {label}.", f"Sketch: {label}"), rng)
            pages = (*text(prompt), sketch_page(drawing.strokes))
        else:
            prompt = instruction("Name this sketch:", ("What is this object?", "Identify this sketch."), rng)
            pages = (*text(prompt), sketch_page(drawing.strokes, targetable=False), *text(label, True))
        return build_document(pages, complete=True, metadata=metadata)

    def manifest(self):
        result={"kind": "formal_documents_v1", "seed": self.seed, "split": self.split,
                "tasks": list(self.tasks), "geometry_version": SCENE_VERSION, "caption_version": CAPTION_VERSION,
                "page_layout": asdict(PageLayoutConfig()), "augmentation": asdict(self.augmentation),
                "font_sha256": hashlib.sha256(Path(self.font_path).read_bytes()).hexdigest(),
                "stories": len(self.stories), "story_source": self.stories.info,
                "raw_source": self.drawings.manifest(), "scene_recipes": self.recipe_count,
                "recipe_source": self.recipes.info if self.recipes else {"seed": self.seed, "lazy": True},
                "instruction_canonical_share": .7}
        if self.object_sampling!='with_replacement':result['object_sampling']=self.object_sampling
        if self.paired_scene_share:
            from src.visual_grounding_scenes import PAIR_VERSION
            result.update(paired_scene_share=self.paired_scene_share, paired_scene_version=PAIR_VERSION)
        return result

    def coverage_limits(self):
        return {'T1':len(self.stories),'O1':len(self.drawings),'O2':len(self.drawings),
                **{task:self.recipe_count for task in ('S1','S2','S3','S4','S5')}}


def build_formal_dataset(*, font_path, raw_dir, story_index, raw_index, scene_index=None,
                         scene_recipes=1000000, seed=42, split="train", augment_text=True,
                         augmentation=None, validation_fraction=.02,object_sampling='with_replacement', paired_scene_share=0., **unused):
    if isinstance(augmentation, dict):
        augmentation = TextAugmentationConfig(**augmentation)
    augmentation = augmentation or TextAugmentationConfig(enabled=augment_text)
    if split == "all":
        raise ValueError("正式RAW索引按train/valid分区；请选择具体分区")
    if scene_index and not (Path(scene_index) / "manifest.json").exists():
        scene_index = str(Path(scene_index) / split)
    return FormalDocumentDataset(VectorFont.from_svg(font_path), IndexedStories(story_index, split),
        IndexedRawSource(raw_dir, raw_index, split), font_path=font_path,
        scene_index=scene_index, scene_recipes=scene_recipes, seed=seed, split=split,
        augmentation=augmentation, validation_fraction=validation_fraction,object_sampling=object_sampling,
        paired_scene_share=paired_scene_share)
